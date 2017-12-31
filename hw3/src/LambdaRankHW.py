__author__ = 'agrotov'

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query
from operator import itemgetter

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

POINTWISE = 'pointwise'
PAIRWISE = 'pairwise'
LISTWISE = 'listwise'

from scipy.special import expit

ALLOWED_MODELS = [POINTWISE, PAIRWISE, LISTWISE]

# TODO: Implement the lambda loss function
def lambda_loss(output, lambdas):
    return output * lambdas


class LambdaRankHW:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, algorithm=POINTWISE):

        if algorithm not in ALLOWED_MODELS:
            raise Exception('Algorithm can only be pointwise, listwise, or pairwise')

        # Type of algorithm to be used by the neural net.
        self.algorithm = algorithm

        # Number of features that are used for an input to the NN.
        self.feature_count = feature_count

        # Symbolic graph that outputs the last layer of the neural network.
        self.output_layer = self.build_model(feature_count, 1, BATCH_SIZE)

        # Dictionary with various function used for the model.
        self.iter_funcs = self.create_functions(self.output_layer)

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 1 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print("input_dim",input_dim, "output_dim",output_dim)
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch, deterministic=True, dtype="float32")


        ###############################################
        # TODO: Change loss function
        # Point-wise loss function (squared error) - comment it out

        if self.algorithm == POINTWISE:
            loss_train = lasagne.objectives.squared_error(output,y_batch)
        else:
            # Pairwise loss function - comment it in
            loss_train = lambda_loss(output,y_batch)
        ###############################################

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)

        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch], output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
        train_func = theano.function(
            [X_batch, y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        )

        print "Finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )

    # TODO: Implement the aggregate (i.e. per document) lambda function
    def lambda_function(self, labels, scores):
        # Should always be the same but as data is noisy
        # I want to make sure
        max_range = min(len(labels), len(scores))

        # Compute in a simplified way
        # taking into account final result of all possible cases
        lambdas = []
        for position in range(max_range):
            up, down, down_count = 0, 0, 0
            for other_position in range(max_range):
                if labels[position] > labels[other_position]:
                    up -= expit(scores[other_position] - scores[position])
                elif labels[position] < labels[other_position]:
                    down_count += 1
                    down += expit(scores[other_position] - scores[position])
            lambdas.append(up - down_count + down)

        return np.array(lambdas, dtype=np.float32)

    def lambda_listwise(self, labels, scores):
        """
        Computes lambdas using the listwise algorithm, i.e.
        taking NDCG into consideration.
        """
        # Again make sure sizes are correct
        max_range = min(len(labels), len(scores))

        # Get sorted sequence, given the scores
        pairs = zip(labels, scores, range(max_range))
        pairs = sorted(pairs, key=itemgetter(1), reverse=True)
        ranking, ranking_scores, indexes = zip(*pairs)

        # Keep track of sorted and normal position
        index = {initial_pos: sorted_pos for sorted_pos, initial_pos in enumerate(indexes)}

        # Get original NDCG
        IDCG = DCG(sorted(ranking, reverse=True), rank=len(ranking))

        def query_ndcg(ranking, size):
            return NDCG(ranking, rank=size, IDCG=IDCG)

        initial_ndcg = NDCG(ranking, rank=max_range, IDCG=IDCG)

        def lambda_pos(position, max_range):
            pos_relevance = labels[position]
            lamb = 0
            sorted_pos = index[position]

            for other_pos in range(max_range):
                if pos_relevance == ranking[other_pos]:
                    continue

                # Swap
                pairs[other_pos], pairs[sorted_pos] = pairs[sorted_pos], pairs[other_pos]

                # Compute delta ndcg
                delta_ndcg = np.fabs(query_ndcg(zip(*pairs)[0], max_range) - initial_ndcg)

                # Compute lambda
                lambda_ij = expit(ranking_scores[other_pos] - pos_relevance) * delta_ndcg
                lamb -= np.sign(pos_relevance - ranking[other_pos]) * lambda_ij

                # Swap back
                pairs[other_pos], pairs[sorted_pos] = pairs[sorted_pos], pairs[other_pos]

            return lamb

        return np.array([lambda_pos(position, max_range) for position in range(max_range)], dtype=np.float32)

    def compute_lambdas_theano(self, query, labels):
        scores = self.score(query).flatten()

        # Pairwise or listwise depending on chosen algorithm
        if self.algorithm == PAIRWISE:
            result = self.lambda_function(labels, scores[:len(labels)])
        elif self.algorithm == LISTWISE:
            result = self.lambda_listwise(labels, scores[:len(labels)])

        return result

    def train_once(self, X_train, query, labels):
        # TODO: Comment out to obtain the lambdas

        if self.algorithm != POINTWISE:
            lambdas = self.compute_lambdas_theano(query, labels)
            lambdas.resize((BATCH_SIZE, ))

            X_train.resize((BATCH_SIZE, self.feature_count), refcheck=False)
        else:
            resize_value = min(BATCH_SIZE, len(labels))
            X_train.resize((resize_value, self.feature_count), refcheck=False)

        # TODO: Comment out (and comment in) to replace labels by lambdas

        if self.algorithm == POINTWISE:
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        else:
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)

        return batch_train_loss

    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in range(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()

                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                batch_train_losses.append(batch_train_loss)
                print 'Finished query ', index

            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

def compute_query_NDCG(query, ranker):
    scores = ranker.score(query).flatten()
    labels = query.get_labels()
    ranking = zip(*sorted(zip(labels, scores), key=itemgetter(1), reverse=True))[0]
    return NDCG(ranking)

def DCG(ranking, rank=10):
    """
    Computes Discounted Cumulative Gain.
    * Chapelle et al. Expected Reciprocal Rank for Graded Relevance.

    Note: Small difference in terms of implementation from homework 1,
    using indexing from 0 and simply account for it in the log.

    Args:
    :param ranking: List of ranking relevance.
    :param rank: Rank to compute DCG at.

    :return: DCG@rank
    """
    return sum([(2 ** ranking[i] - 1) / np.log2(i + 2) for i in range(rank)])

def NDCG(ranking, rank=10, IDCG=None):
    """
    Computes Normalized Discounted Cumulative Gain.
    """
    if not IDCG:
        IDCG = DCG(sorted(ranking, reverse=True), rank=len(ranking))

    return DCG(ranking, rank=rank) / IDCG if IDCG != 0 else 0

def run_exp():
    ### CHANGE ALGORITHM HERE ###
    ## Possible values: POINTWISE, PAIRWISE, LISTWISE
    ALGORITHM = PAIRWISE # Change here for other algos
    FEATURES = 64
    EPOCHS = 5
    num_folds = 5
    NDCG_AVG = []
    folds = []
    for fold in range(1, num_folds + 1):
        # Ranker for this cross-validation.
        ranker = LambdaRankHW(FEATURES, algorithm=ALGORITHM)

        # Fold NDCG scores
        fold_scores = []

        for cross_fold in range(1, num_folds + 1):
            if fold == cross_fold:
                continue

            # Current fold taining queries
            training_queries = query.load_queries("HP2003/Fold%d/train.txt" % cross_fold, FEATURES)
            ranker.train_with_queries(training_queries, EPOCHS)

            fold_scores.append(np.mean([compute_query_NDCG(q, ranker) for q in training_queries]))

        # Load test queries
        test_queries = query.load_queries("HP2003/Fold%d/test.txt" % fold, FEATURES)

        # Compute and add NDCG on test set
        NDCG_AVG.append(np.mean([compute_query_NDCG(q, ranker) for q in test_queries]))

        # Also store NDCG scores on fold to plot them
        folds.append(fold_scores)

    # Save and compute average over all folds
    list_file_name = ALGORITHM + "_NDCGS.npy"
    average_file_name = ALGORITHM + "_average_NDCG.npy"
    all_fold_scores = ALGORITHM + "_allscores.npy"

    np.save(list_file_name, np.array(NDCG_AVG))
    np.save(all_fold_scores, np.array(folds))

    total_average = np.average(NDCG_AVG)
    np.save(average_file_name, total_average)


if __name__ == "__main__":
    run_exp()