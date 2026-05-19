# 1. Base solida con Node.js preinstallato
FROM node:24-slim

# 2. Installa il router in modo permanente nell'immagine
RUN npm install -g @musistudio/claude-code-router

# 3. Comunica a Docker di esporre la porta corretta
EXPOSE 3456

# 4. Avvia il router all'accensione del container
CMD ["sh", "-c", "ccr start && tail -f /dev/null"]