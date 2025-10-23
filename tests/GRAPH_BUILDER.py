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

if __name__ == "__main__":
    test_process_llm_response()
