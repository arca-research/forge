# steps to create a new extraction domain
1. Create a new file in `_templates/` named after your domain. Your own higher-level directory can also be used, just specify it in `GraphConfig` of `config.py` (`template_directory`)
   Example: `_templates/news_article.py`
2. Copy the structure & contents from `_templates/base.py`:
```python
   ENTITY_TYPES = ["ENTITY_TYPE_1", "ENTITY_TYPE_2"]
   
   EXTRACTION_TEMPLATE = """
   [instructions and examples]
   """
```
3. Define `ENTITY_TYPES` with your entity types (what you want to extract).
4. In `EXTRACTION_TEMPLATE`:
   - Provide clear extraction instructions
   - Include 1-2 example documents from your domain
   - Show the expected output format for each example
   - Include placeholders: `{entity_types}`, `{document}`, `{context}`
5. Register your domain in `GraphConfig.extraction_domains`:
```python
   extraction_domains: list[str] = field(default_factory=lambda: [
       "base",
       "news_article",  # your new domain
   ])
```