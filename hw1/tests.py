class Query(object):
	def __init__(self, session_id, time_passed, query_id, region_id, documents):
		self.query_id = query_id
		self.session_id = session_id
		self.time_passed = time_passed
		self.region_id = region_id
		self.documents = documents

	def add_documents(self, documents):
		self.documents.append(documents)


class Click(object):
	def __init__(self, session_id, time_passed, document_id):
		self.session_id = session_id
		self.document_id = document_id
		self.time_passed = time_passed

data = []
with open('YandexRelPredChallenge.txt') as f:
	for row in f:
		row_data = row.split('\t')
		row_data = [col.strip() for col in row_data]
		if row_data[2] == 'C':
			click = Click(session_id=row_data[0], time_passed=row_data[1],
			              document_id=row_data[3])
			data.append((row_data[0], click))
		else:
			docs = [row_data[i] for i in range(5, len(row_data))]
			query = Query(session_id=row_data[0], time_passed=row_data[1],
			              query_id=row_data[3], region_id=row_data[4], documents=docs)
			data.append((row_data[0], query))


# Test to see if in all session where a query id is repeated,
# if the list of document is different, it has to be completely different
# otherwise it has to be completely identical, in which case we simply have a duplicate
# and the user just hit submit again.

# Counter for identical queries
same_documents = 0
same_doc_sessions = {}

# New documents, i.e. pagination
paginated_documents = 0
paginated_sessions = {}

# This should be 0 always, same query in same session
# only some documents different.
problematic_queries = 0
prob_sessions = {}

index = 0
while index < len(data):
	element = data[index]
	curr_session = element[0]
	curr_index = index
	queries = []
	query_positions = {}
	while curr_session == element[0]:
		if isinstance(data[curr_index][1], Query):
			query = data[curr_index][1]
			if query.query_id in query_positions:
				# Found a duplicate, it is either the exact same
				# query (i.e. same documents), or the next page of
				# that query. Test to see which one
				current_position = curr_index
				last_query, last_position = query_positions[query.query_id]

				# Check if all documents are the same
				all_same = True
				for j in range(10):
					if last_query.documents[j] != query.documents[j]:
						all_same = False
						break

				if all_same:
					same_documents += 1
					same_doc_sessions[curr_session] = True

				all_different = True
				for j in range(10):
					if last_query.documents[j] == query.documents[j]:
						all_different = False
						break

				if all_different:
					paginated_documents += 1
					paginated_sessions[curr_session] = True

				if not all_same and not all_different:
					problematic_queries += 1
					prob_sessions[curr_session] = True
			else:
				queries.append(query)
				query_positions[query.query_id] = (query, curr_index)
		curr_index += 1
		if curr_index < len(data):
			curr_session = data[curr_index][0]
		else:
			curr_session = None
	index = curr_index