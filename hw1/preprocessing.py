class Query(object):
	def __init__(self, session_id, time_passed, query_id, region_id, documents):
		self.query_id = query_id
		self.session_id = session_id
		self.time_passed = time_passed
		self.region_id = region_id
		self.documents = documents
		self.clicks = []

	def has_click(self, click_document_id):
		# Determines if the clicked document is in this query
		# and returns it's rank.
		for rank, doc in enumerate(self.documents):
			if doc == click_document_id:
				return rank

		return -1

	def add_documents(self, documents):
		self.documents.append(documents)

	def has_document(self, document_id):
		for doc in self.documents:
			if document_id == doc:
				return True

		return False

	def identical_queries(self, query_row):
		# Compares a query object with a query row
		# from the Yandex log file.
		# Returns True if they share the same id and all documents are identical.
		if self.query_id == query_row[3]:
			for i in range(10):
				if self.documents[i] != query_row[5 + i]:
					return False
			return True
		return False

	def distinct_queries(self, query_row):
		# Compares a query object with a query row
		# from the Yandex log file.
		# Return True if they share the same id, but all 10 documents are different
		# (i.e. one is the next page of the other)
		if self.query_id == query_row[3]:
			for i in range(10):
				if self.documents[i] == query_row[5 + i]:
					return False
			return True
		return False


class Click(object):
	def __init__(self, session_id, time_passed, document_id, query_id, rank):
		self.session_id = session_id
		self.document_id = document_id
		self.time_passed = time_passed
		self.query_id = query_id
		self.rank = rank

def getYandexData():
		data = []
		with open('YandexRelPredChallenge.txt') as f:
				for row in f:
						row_data = row.split('\t')
						row_data = [col.strip() for col in row_data]
						data.append(row_data)
		return data

yandex_data = getYandexData()
index = 0
query_session_counter = 0
all_queries = []
while index < len(yandex_data):
	print(index)
	# Current row
	datapoint = yandex_data[index]

	# Current user session id
	session = datapoint[0]

	# Process current sessions queries and clicks.
	session_queries = []
	session_unique_queries = {}

	# Iterate through current session
	sess = session
	i = index
	while sess == session:
		row_datapoint = yandex_data[i]

		# Found a query
		if row_datapoint[2] == 'Q':
			query_id = row_datapoint[3]

			# We have never seen this query before
			if not query_id in session_unique_queries:
				# Create query object
				query = Query(session_id=query_session_counter,
											time_passed=row_datapoint[1],
											query_id=row_datapoint[3],
											region_id=row_datapoint[4],
											documents=[row_datapoint[i] for i in range(5, len(row_datapoint))])
				query_session_counter += 1

				# Store query as newest query
				session_queries.append(query)
				session_unique_queries[query_id] = query
			else:
				# We have seen this query before, handle corner cases

				# Same query id, determine difference if there is one.
				prev_query = session_unique_queries[query_id]

				# Queries are identical
				if prev_query.identical_queries(row_datapoint):
					# Simply move the position of the query to the top of the list.
					session_queries.pop(session_queries.index(prev_query))
					session_queries.append(prev_query)

				elif prev_query.distinct_queries(row_datapoint):
					# Same query, different documents, simply append them to the end of the list
					# and move query to front.
					prev_query.add_documents(
						[row_datapoint[i] for i in range(5, len(row_datapoint))])
					session_queries.pop(session_queries.index(prev_query))
					session_queries.append(prev_query)
				else:
					# Same documents, only some documents are the same.
					# Create a new query object from the current row.
					# Create query object
					query = Query(session_id=query_session_counter,
												time_passed=row_datapoint[1],
												query_id='###',  # Query id doesn't matter for this case.
												region_id=row_datapoint[4],
												documents=[row_datapoint[i] for i in
																	 range(5, len(row_datapoint))])
					query_session_counter += 1

					# Only store query object as newest query.
					session_queries.append(query)
		else:
			# Process click
			# Find latest query where click is present.
			document_clicked = row_datapoint[3]

			# Start from latest query
			for p_query in session_queries[::-1]:
				r = p_query.has_click(document_clicked)
				if r != -1:
					# Found click
					click = Click(session_id=p_query.session_id,
												time_passed=row_datapoint[1],
												document_id=document_clicked,
												query_id=p_query.query_id,
												rank=r)
					# Store click
					p_query.clicks.append(click)
					break
		i += 1
		print('New i is', i)
		if i < len(yandex_data):
			sess = yandex_data[i][0]
		else:
			sess = None

	all_queries.extend(session_queries)
	index = i

print(len(all_queries))