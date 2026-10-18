"""Single-worker server using Render's runtime PORT."""
import os

import uvicorn

uvicorn.run("legal_assistant.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
            workers=1, proxy_headers=False, access_log=False)
