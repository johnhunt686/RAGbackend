# RAG Chunk Testing
## Testing round 1
before anything complex can occur we must first configure the chunking behavior to be ideal.
I've started with 1 document with varied information
1706.03762v7.pdf (Attention Is All You Need)
this reasearch paper has many tables and lots of specific information

I began with some simple tests on the following settings
CHUNK_MAX_TOKENS = 512
TOP_K = 5
this produces five 512 token long sections scored based on their distance in the vectorDB, this passes through llamaIndex and becomes a score. As you can see in the results it does a good job of fetching the information from the relevant documents and even provides an source of the information. This is a strong start and already gives our model a very high chance of answering questions correctly.

