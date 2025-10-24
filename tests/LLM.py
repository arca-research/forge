from .common import TEST_LOG,  LLM_CONFIG
from ..forge.llm import SyncLLM, AsyncLLM

def test_openai():
    return

def test_local():
    return

def test_openrouter():
    return

def test_error_retry():
    _api_key = LLM_CONFIG.api_key
    _model = LLM_CONFIG.sync_model
    llm = SyncLLM(backend="openai", model=_model, api_key=_api_key, url="null", retries=3)
    llm.set_system("You are a helpful assistant.")
    msg = llm.run("Hello")
    print(msg)
    return


if __name__ == "__main__":
    test_error_retry()