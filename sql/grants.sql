-- Replace this application ID if the job run-as principal changes.
GRANT USE CATALOG ON CATALOG education_rag TO `32330b70-b665-4733-8da9-4906dab67168`;
GRANT USE SCHEMA ON SCHEMA education_rag.bronze TO `32330b70-b665-4733-8da9-4906dab67168`;
GRANT SELECT ON TABLE education_rag.bronze.markdown_documents TO `32330b70-b665-4733-8da9-4906dab67168`;
GRANT USE SCHEMA, CREATE TABLE, MODIFY, SELECT ON SCHEMA education_rag.silver TO `32330b70-b665-4733-8da9-4906dab67168`;
