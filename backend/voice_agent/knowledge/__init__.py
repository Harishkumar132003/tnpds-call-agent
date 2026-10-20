"""Document knowledge base: extract, chunk, embed and store text in Qdrant.

Used by the portal for uploads and retrieval tests. The agent does not use it
yet; when it does, :class:`voice_agent.knowledge.service.KnowledgeService`
is the entry point so the pipeline never talks to Qdrant or OpenAI directly.
"""
