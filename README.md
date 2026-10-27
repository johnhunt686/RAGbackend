# requirements
- cloudflared (runs at startup)
- ollama (runs at startup)
- suitable models
# install
- symlink reasearch folder
- modifty rag-api.service
- add rag-api.service to /etc/systemd/system/
- populate reasearch folder
- run ingest (optional)
# run
- while services can be manually invoked they should start with the system according to the rag-api.service file
- the rag api runs independantly of the apps database, though both share the reasearch folder. As of 10/26/26 the systems are completely independent.






test queries
curl -s localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-dev" -H 'Content-Type: application/json' \
  -d '{"model":"rag-docs","messages":[{"role":"user","content":"What optimizer was used?"}]}'