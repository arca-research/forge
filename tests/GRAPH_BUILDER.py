from .common import TEST_LOG, GRAPH_CONFIG, GRAPH_BUILDER

def test_process_llm_response():
    sample_llm_response_green = """
    ("entity"{tuple_delimiter}Republic of Arcania{tuple_delimiter}GEO{tuple_delimiter}Country in which the Ministry of Urban Mobility and the city of Sankt Rúna are located)
    {record_delimiter}
    ("relationship"{tuple_delimiter}Transparency Watch Europa{tuple_delimiter}Ministry of Urban Mobility of the Republic of Arcania{tuple_delimiter}Filed a procurement-integrity complaint citing conflicts from overlapping board memberships at proposed subcontractors)
    {completion_delimiter}
    """.strip().format(
        tuple_delimiter=GRAPH_CONFIG.tuple_delimiter, record_delimiter=GRAPH_CONFIG.record_delimiter, completion_delimiter=GRAPH_CONFIG.completion_delimiter
    )

    TEST_LOG.info("---GREEN---")
    entities, relationships = GRAPH_BUILDER._process_llm_response(sample_llm_response_green)
    TEST_LOG.info("entities: %s", entities)
    TEST_LOG.info("relationships: %s", relationships)

    sample_llm_response_red = """
    ("entity"{tuple_delimiter}Republic of Arcania{tuple_delimiter}GEO)
    {record_delimiter}
    ("relationship"{tuple_delimiter}Transparency Watch Europa{tuple_delimiter}Ministry of Urban Mobility of the Republic of Arcania{tuple_delimiter}Filed a procurement-integrity complaint citing conflicts from overlapping board memberships at proposed subcontractors)
    {completion_delimiter}
    """.strip().format(
        tuple_delimiter=GRAPH_CONFIG.tuple_delimiter, record_delimiter=GRAPH_CONFIG.record_delimiter, completion_delimiter=GRAPH_CONFIG.completion_delimiter
    )

    TEST_LOG.info("---RED---")
    entities, relationships = GRAPH_BUILDER._process_llm_response(sample_llm_response_red)
    TEST_LOG.info("entities: %s", entities)
    TEST_LOG.info("relationships: %s", relationships)


def test_progress_bar():
    from ..src.util import print_progress_bar
    import time
    docs = ["a", "b", "c"]
    total = len(docs)
    print_progress_bar(0, total)
    for i in range(0, len(docs), 1):
        batch = docs[i : i + 1]
        for j, doc in enumerate(batch):
            current = i + j + 1
            time.sleep(3)
            TEST_LOG.info(f'{doc}')
            print_progress_bar(current, total)
    

if __name__ == "__main__":
    # test_process_llm_response()
    test_progress_bar()
