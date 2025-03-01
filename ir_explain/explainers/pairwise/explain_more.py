import itertools
import os

import pandas as pd
from ir_explain.utils.pairwise_utils import *


def calculate_avg_distance(occurrences):
    distances = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
    return sum(distances) / len(distances) if distances else 0

def calculate_term_discrimination_values(index_path):
            term_doc_freq = {}
            total_docs = 0

            for filename in os.listdir(index_path):
                if filename.endswith('.txt'):
                    total_docs += 1
                    with open(os.path.join(index_path, filename), 'r') as file:
                        document = file.read()
                        terms = set(document.split())
                        for term in terms:
                            if term in term_doc_freq:
                                term_doc_freq[term] += 1
                            else:
                                term_doc_freq[term] = 1

            term_discrimination_values = {term: 1.0 / freq for term, freq in term_doc_freq.items()}
            return term_discrimination_values

class ExplainMore:

    class TFC1:

        def explain(query, document1, document2,index_path):

            if abs(len(document1) - len(document2)) >= 0.1 * max(len(document1),len(document2)):
                print("Lengths of documents not similar")
                return 0

            def term_frequency(term, document):
              return document.split().count(term)

            query_terms = query.split()

            doc1_tf = sum(term_frequency(term, document1) for term in query_terms)
            doc2_tf = sum(term_frequency(term, document2) for term in query_terms)

            print(f"Term Frequency of query terms in document1 is {doc1_tf}")
            print(f"Term Frequency of query terms in document2 is {doc2_tf}")
            return 0
    

    class TFC3:

        def explain(query, document1, document2,index_path):
            query_terms = query.split()
            query_term_set = set(query_terms)

            td_value = 1.0
            for term in query_terms:
                td_value = calculate_term_discrimination_values(index_path).get(term, 1.0)

            doc1_words = document1.split()
            doc2_words = document2.split()

            if len(doc1_words) != len(doc2_words):
                print("Length of docs not the same")
                return 0

            c_q1_D1 = doc1_words.count(query_terms[0])
            c_q2_D1 = doc1_words.count(query_terms[1])
            c_q1_D2 = doc2_words.count(query_terms[0])
            c_q2_D2 = doc2_words.count(query_terms[1])

            if not (c_q1_D1 == c_q1_D2 + c_q2_D2 and c_q2_D1 == 0 and c_q1_D2 == 0 and c_q2_D2 == 0):
                print("The given documents do not satisfy the axiom conditions.")
                return 0

            S_Q_D1 = td_value * (c_q1_D1 + c_q2_D1)
            S_Q_D2 = td_value * (c_q1_D2 + c_q2_D2)

            print(f"Score of document 1: {S_Q_D1}")
            print(f"Score of document 2:  {S_Q_D2}")
            return 0    

    class TDC:

        def explain(query, document1, document2, index_path):
            query_terms = query.split()
            if len(query_terms) != 2:
                print("Axiom Conditions not satisfied")
                return 0
            
            q1, q2 = query_terms
            
            document1_words = set(document1.split())
            document2_words = set(document2.split())

            if q1 in document1_words and q2 not in document1_words and q2 in document2_words and q1 not in document2_words:
                td_q1 = calculate_term_discrimination_values(index_path).get(q1, 1.0)
                td_q2 = calculate_term_discrimination_values(index_path).get(q2, 1.0)
                
                print(f"Score of document 1: {td_q1}")
                print(f"Score of document 2: {td_q2}")
            else:
                print("Axiom Conditions not satisfied")
            return 0
            
    class M_TDC:

        def explain(query, document1, document2,index_path):
            query_terms = query.split()
            if len(query_terms) != 2:
                print("Axiom Conditions not satisfied")
                return 0
            
            q1, q2 = query_terms
            
            c_w1_d1 = document1.split().count(q1)
            c_w2_d1 = document1.split().count(q2)
            c_w1_d2 = document2.split().count(q1)
            c_w2_d2 = document2.split().count(q2)

            if c_w1_d1 == c_w2_d2 and c_w2_d1 == c_w1_d2:
                td_q1 = calculate_term_discrimination_values(index_path).get(q1, 1.0)
                td_q2 = calculate_term_discrimination_values(index_path).get(q2, 1.0)

                print(f"Score of document 1: {td_q1}")
                print(f"Score of document 2: {td_q2}")
                
            else:
                print("Axiom Conditions not satisfied")
                    
    class PROX1:

        def calculate_avg_distance(occurrences):
              if len(occurrences) < 2:
                  return float('inf')  
              distances = [occurrences[i + 1] - occurrences[i] for i in range(len(occurrences) - 1)]
              return sum(distances) / len(distances)

        def explain(query, document1, document2,index_path):
              query_words = query.split()

              words_doc1 = document1.split()
              words_doc1 = [word.replace('.', '') for word in words_doc1]
              words_doc2 = document2.split()
              words_doc2 = [word.replace('.', '') for word in words_doc2]

              term_pairs = list(itertools.combinations(query_words, 2))

              avg_distances = {pair: [] for pair in term_pairs}
              term_frequencies = {word: {'Document 1': 0, 'Document 2': 0} for word in query_words}

              for term1, term2 in term_pairs:
                  occurrences_doc1_term1 = [i for i, w in enumerate(words_doc1) if w == term1]
                  occurrences_doc1_term2 = [i for i, w in enumerate(words_doc1) if w == term2]

                  occurrences_doc2_term1 = [i for i, w in enumerate(words_doc2) if w == term1]
                  occurrences_doc2_term2 = [i for i, w in enumerate(words_doc2) if w == term2]

                  avg_distance_doc1 = calculate_avg_distance(occurrences_doc1_term1) + calculate_avg_distance(occurrences_doc1_term2)
                  avg_distance_doc2 = calculate_avg_distance(occurrences_doc2_term1) + calculate_avg_distance(occurrences_doc2_term2)

                  avg_distances[(term1, term2)].extend([avg_distance_doc1, avg_distance_doc2])

              for word in query_words:
                  term_frequencies[word]['Document 1'] = words_doc1.count(word)
                  term_frequencies[word]['Document 2'] = words_doc2.count(word)

              rows = []
              for word in query_words:
                  row = [f'tf({word})', term_frequencies[word]['Document 1'], term_frequencies[word]['Document 2']]
                  rows.append(row)

              total_avg_dist_doc1 = 0
              total_avg_dist_doc2 = 0

              for term_pair, distances in avg_distances.items():
                  row = [f'avg_dist({term_pair[0]}, {term_pair[1]})', distances[0], distances[1]]
                  rows.append(row)
                  total_avg_dist_doc1 += distances[0]
                  total_avg_dist_doc2 += distances[1]

              num_pairs = len(term_pairs)
              rows.append(['num pairs', num_pairs, num_pairs])
              rows.append(['Total_avg_dist', total_avg_dist_doc1 / num_pairs, total_avg_dist_doc2 / num_pairs])

              df = pd.DataFrame(rows, columns=['Metric', 'Document 1', 'Document 2'])

              return df

    class PROX2:

        def explain(query, document1, document2,index_path):

            query_words = query.split()

            words_doc1 = document1.split()
            words_doc1 = [word.replace('.', '') for word in words_doc1]
            words_doc2 = document2.split()
            words_doc2 = [word.replace('.', '') for word in words_doc2]

            first_occurrences = {}

            for term in query_words:
                index = next((i for i, word in enumerate(words_doc1) if word == term), None)
                first_occurrences[f'Document 1 - {term}'] = index

            for term in query_words:
                index = next((i for i, word in enumerate(words_doc2) if word == term), None)
                first_occurrences[f'Document 2 - {term}'] = index

            for key, value in first_occurrences.items():
                print(f"{key}: {value}")
            return 0

    class PROX3:

        def explain(query, document1, document2,index_path):

            index_doc1 = document1.find(query)
            index_doc2 = document2.find(query)

            print(f"First occurrence of the query in Document 1: {index_doc1 if index_doc1 != -1 else 'Not present'}")
            print(f"First occurrence of the query in Document 2: {index_doc2 if index_doc2 != -1 else 'Not present'}")
            return 0

    class PROX4:

        def explain(query, doc1, doc2,index_path):
            query_terms = set(query.split())

            def smallest_span(document):
                words = document.split()
                term_positions = {term: [] for term in query_terms}

                for idx, word in enumerate(words):
                    if word in query_terms:
                        term_positions[word].append(idx)

                min_span_length = float('inf')
                min_span_non_query_count = float('inf')
                min_span = []

                for term in term_positions:
                    for start_pos in term_positions[term]:
                        end_pos = start_pos
                        valid = True
                        for other_term in term_positions:
                            if other_term != term:
                                if term_positions[other_term]:
                                    closest_pos = min(term_positions[other_term], key=lambda x: abs(x - start_pos))
                                    end_pos = max(end_pos, closest_pos)
                                else:
                                    valid = False
                                    break
                        
                        if valid and end_pos - start_pos + 1 < min_span_length:
                            min_span = words[start_pos:end_pos + 1]
                            min_span_length = len(min_span)
                            min_span_non_query_count = sum(1 for word in min_span if word not in query_terms)

                return min_span_non_query_count

            def calculate_gap(document):
                min_span_non_query_count = smallest_span(document)
                words = document.split()
                gap_frequency = words.count(str(min_span_non_query_count))

                return (min_span_non_query_count, gap_frequency)

            gap1 = calculate_gap(doc1)
            gap2 = calculate_gap(doc2)

            print(f"Gap in document 1: {gap1}")
            print(f"Gap in document 2: {gap2}")
            return 0

    class PROX5:

        def explain(query, doc1, doc2,index_path):

          query_terms = query.split()

          def find_positions(term, document):
              positions = []
              words = document.split()
              for idx, word in enumerate(words):
                  if word == term:
                      positions.append(idx)
              return positions

          def smallest_span_around(term_positions, all_positions, num_terms):
              min_span = float('inf')
              for pos in term_positions:
                  spans = []
                  for i in range(num_terms):
                      term_pos = all_positions[i]
                      if term_pos:
                          distances = [abs(pos - p) for p in term_pos]
                          spans.append(min(distances))
                  if len(spans) == num_terms:
                      min_span = min(min_span, max(spans) - min(spans) + 1)
              return min_span

          def average_smallest_span(document):
              all_positions = [find_positions(term, document) for term in query_terms]
              total_span = 0
              count = 0

              for i, term_positions in enumerate(all_positions):
                  if term_positions:
                      span = smallest_span_around(term_positions, all_positions, len(query_terms))
                      if span < float('inf'):
                          total_span += span
                          count += 1

              return total_span / count if count > 0 else float('inf')

          span1 = average_smallest_span(doc1)
          span2 = average_smallest_span(doc2)

          print(f"average smallest text span across all occurrences query terms in doc1 is {span1}")
          print(f"average smallest text span across all occurrences query terms in doc2 is {span2}")
          return 0

    class LNC1:

        def explain(query, document1, document2,index_path):

            query_words = query.split()

            count_query_terms_doc1 = sum(1 for word in query_words if word in document1)
            count_query_terms_doc2 = sum(1 for word in query_words if word in document2)

            print(f"Number of query terms document 1: {count_query_terms_doc1}")
            print(f"Number of query terms document 2: {count_query_terms_doc2}")
            print()
            print(f"Length of document1: {len(document1)}")
            print(f"Length of document2: {len(document2)}")
            return 0

    class TF_LNC:

        def explain(query, document1, document2,index_path):

            query_words = set(query.split())
            document1_words = set(document1.split())
            document2_words = set(document2.split())

            common_words1 = query_words.intersection(document1_words)
            common_words2 = query_words.intersection(document2_words)


            words1 = document1.split()
            words2 = document2.split()

            filtered_words1 = [word for word in words1 if word not in common_words1]
            filtered_words2 = [word for word in words2 if word not in common_words2]

            new_doc1 = ' '.join(filtered_words1)
            new_doc2 = ' '.join(filtered_words2)

            max_len = max(len(new_doc1), len(new_doc2))
            tolerance = 0.1 * max_len

            if abs(len(new_doc1) - len(new_doc2)) > tolerance:
                print("Documents are not of approximately equal length")
            else:
                print(f"Query terms in document1: {len(common_words1)}")
                print(f"Query terms in document2: {len(common_words2)}")
            return 0

    class LB1:

        def explain(query, document1, document2,index_path):

            query_terms = set(query.lower().split())
            doc1_terms = set(document1.lower().split())
            doc2_terms = set(document2.lower().split())

            unique_to_doc1 = [term for term in query_terms if term in doc1_terms and term not in doc2_terms]
            unique_to_doc2 = [term for term in query_terms if term in doc2_terms and term not in doc1_terms]

            print(f"Query terms present in document 1 but not in document 2 {unique_to_doc1}")
            print(f"Query terms present in document 2 but not in document 1 {unique_to_doc2}")
            return 0

    class STMC1:

        def explain(query, document1, document2,index_path):

          similarity_doc1, similarity_doc2 = wordnet_similarity(query, document1, document2)

          print(f"similarity score of doc1 with query terms is {similarity_doc1}")
          print(f"similarity score of doc2 with query terms is {similarity_doc2}")
          return 0
    
    class STMC2:

        def explain(query, document1, document2,index_path):
            query_terms = set(query.split())
            
            def calculate_term_frequencies(document):
                words = document.split()
                term_freq = {}
                for word in words:
                    if word in term_freq:
                        term_freq[word] += 1
                    else:
                        term_freq[word] = 1
                return term_freq
            
            def find_max_similar_term(query_terms, document):
                doc_words = set(document.split())
                non_query_terms = doc_words - query_terms
                max_sim = 0
                max_term = None
                max_query_term = None
                for query_term in query_terms:
                    for word in doc_words:
                        sim = w_sim2(query_term, word)
                        if sim > max_sim:
                            max_sim = sim
                            max_term = word
                            max_query_term = query_term
                return max_term, max_query_term, max_sim
            
            term_freq_d1 = calculate_term_frequencies(document1)
            term_freq_d2 = calculate_term_frequencies(document2)

            max_term_d1, max_query_term_d1, max_sim_d1 = find_max_similar_term(query_terms, document1)
            max_term_d2, max_query_term_d2, max_sim_d2 = find_max_similar_term(query_terms, document2)

            if max_sim_d1 > max_sim_d2:
                max_term = max_term_d1
                max_query_term = max_query_term_d1
            else:
                max_term = max_term_d2
                max_query_term = max_query_term_d2

            len_d1 = len(document1.split())
            len_d2 = len(document2.split())

            tf_t_d1 = term_freq_d1.get(max_term, 0)
            tf_t_d2 = term_freq_d2.get(max_term, 0)
            tf_t0_d1 = term_freq_d1.get(max_query_term, 0)

            ratio = len_d2 / len_d1
            tf_ratio = tf_t_d2 / (tf_t0_d1 if tf_t0_d1 != 0 else 1)

            print(f"Maximally similar non-query term (t) in either document: '{max_term}'")
            print(f"Query term (t0) maximally similar to t: '{max_query_term}'\n")
            
            print(f"tf_t_d1: Frequency of '{max_term}' in document1: {tf_t_d1}")
            print(f"tf_t_d2: Frequency of '{max_term}' in document2: {tf_t_d2}")
            print(f"tf_t0_d1: Frequency of '{max_query_term}' in document1: {tf_t0_d1}\n")
            
            print(f"Length of Document 1 (|D1|): {len_d1}")
            print(f"Length of Document 2 (|D2|): {len_d2}")
            print(f"Ratio |D2| / |D1|: {ratio}")
            print(f"Term Frequency Ratio tf(t, D2) / tf(t0, D1): {tf_ratio}\n")

    class AND:

        def explain(query, document1, document2,index_path):

            query_terms = set(query.lower().split())
            doc1_terms = set(document1.lower().split())
            doc2_terms = set(document2.lower().split())

            if query_terms.issubset(doc1_terms):
                print("All query terms present in document 1")
            if query_terms.issubset(doc2_terms):
                print("All query terms present in document 2")

            not_in_doc1 = query_terms - doc1_terms
            if not_in_doc1:
                print("Query terms not present in document 1:", not_in_doc1)
            else:
                print("All query terms are present in document 1")

            not_in_doc2 = query_terms - doc2_terms
            if not_in_doc2:
                print("Query terms not present in document 2:", not_in_doc2)
            else:
                print("All query terms are present in document 2")
            return 0

    class REG:

        def explain(query, document1, document2,index_path):
          
          query_terms = query.lower().split()

          most_similar_term = get_most_similar_term(query_terms)

          if not most_similar_term:
              print("There is no term in query similar to other query terms")
              return 0
          
          print(f"Query term most similar to other terms is {most_similar_term}")
          all_texts = [query] + [document1, document2]

          vectorizer = CountVectorizer()
          term_frequency_matrix = vectorizer.fit_transform(all_texts)

          most_similar_term_index = vectorizer.vocabulary_[most_similar_term]
          doc1_term_frequency = term_frequency_matrix[-2, most_similar_term_index]
          doc2_term_frequency = term_frequency_matrix[-1, most_similar_term_index]

          print(f"Term frequency of {most_similar_term} in document 1 is : {doc1_term_frequency}")
          print(f"Term frequency of {most_similar_term} in document 2 is : {doc2_term_frequency}")
          return 0

    class DIV:

        def explain(query, document1, document2,index_path):

            query_terms = set(query.lower().split())
            doc1_terms = set(document1.lower().split())
            doc2_terms = set(document2.lower().split())

            jaccard_coefficient_doc1 = len(query_terms.intersection(doc1_terms)) / len(query_terms.union(doc1_terms))
            jaccard_coefficient_doc2 = len(query_terms.intersection(doc2_terms)) / len(query_terms.union(doc2_terms))

            print(f"Jaccard Co-efficient of doc1 is:{jaccard_coefficient_doc1}")
            print(f"Jaccard Co-efficient of doc2 is:{jaccard_coefficient_doc2}")
            return 0
