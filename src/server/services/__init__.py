"""Server service layer: domain logic kept out of the HTTP/WS routes.

Contract: plain Python results; routes own FastAPI request/response shapes.
Dependencies: src.server.state and the agent legs.
Used by: src.server.routes modules.
"""
