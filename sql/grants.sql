-- Run as the catalog/schema owner. Replace the principal with the Databricks
-- service-principal display name or application ID registered in the workspace.
GRANT USE CATALOG ON CATALOG education_rag TO `github_markdown_autoloader`;
GRANT USE SCHEMA ON SCHEMA education_rag.bronze TO `github_markdown_autoloader`;
GRANT SELECT ON TABLE education_rag.bronze.markdown_documents TO `github_markdown_autoloader`;
GRANT USE SCHEMA ON SCHEMA education_rag.silver TO `github_markdown_autoloader`;
GRANT CREATE TABLE ON SCHEMA education_rag.silver TO `github_markdown_autoloader`;

-- Run after the first successful setup_tables task if the tables were created
-- by a different owner.
GRANT SELECT, MODIFY ON TABLE education_rag.silver.page_classification_windows TO `github_markdown_autoloader`;
GRANT SELECT, MODIFY ON TABLE education_rag.silver.fine_grained_pages TO `github_markdown_autoloader`;
