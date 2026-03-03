from flask import Blueprint, jsonify, request

from app import services
from app.data_store import DataStore
from app.llm_providers import available_providers

bp = Blueprint("api", __name__, url_prefix="/api")


def _validate_client(client: str):
    """Return (client, error_response) — error_response is None if valid."""
    providers = available_providers()
    if client not in providers:
        return None, (
            jsonify({"error": f"Unknown client '{client}'. Available: {providers}"}),
            400,
        )
    return client, None


def _service_error_response(exc: Exception):
    """Map service-layer exceptions to HTTP responses."""
    if isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 422
    if isinstance(exc, LookupError):
        return jsonify({"error": str(exc)}), 404
    if isinstance(exc, RuntimeError):
        return jsonify({"error": str(exc)}), 409
    raise exc


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        result = services.upload_document(f.read(), f.filename)
    except Exception as exc:
        return _service_error_response(exc)

    return jsonify(result), 201


# ---------------------------------------------------------------------------
# POST /api/files/<file_id>/index
# ---------------------------------------------------------------------------

@bp.route("/files/<file_id>/index", methods=["POST"])
def index_file(file_id: str):
    body = request.get_json(silent=True) or {}
    client = body.get("client", "claude").lower()
    client, err = _validate_client(client)
    if err:
        return err

    try:
        result = services.index_document(file_id, client)
    except Exception as exc:
        return _service_error_response(exc)

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# POST /api/files/<file_id>/extract  (on-demand LLM extraction)
# ---------------------------------------------------------------------------

@bp.route("/files/<file_id>/extract", methods=["POST"])
def llm_extract(file_id: str):
    body = request.get_json(silent=True) or {}
    client = body.get("client", "openai").lower()
    client, err = _validate_client(client)
    if err:
        return err

    try:
        result = services.llm_extract_document(file_id, client)
    except Exception as exc:
        return _service_error_response(exc)

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

@bp.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    file_id = body.get("file_id")
    user_message = body.get("message", "").strip()
    conv_id = body.get("conv_id")
    client = body.get("client", "claude").lower()

    if not file_id:
        return jsonify({"error": "'file_id' is required"}), 400
    if not user_message:
        return jsonify({"error": "'message' is required"}), 400

    client, err = _validate_client(client)
    if err:
        return err

    try:
        result = services.chat_with_document(file_id, user_message, conv_id, client)
    except Exception as exc:
        return _service_error_response(exc)

    return jsonify(result)


# ---------------------------------------------------------------------------
# GET /api/providers
# ---------------------------------------------------------------------------

@bp.route("/providers", methods=["GET"])
def list_providers():
    return jsonify({"providers": available_providers()})


# ---------------------------------------------------------------------------
# GET /api/files
# ---------------------------------------------------------------------------

@bp.route("/files", methods=["GET"])
def list_files():
    return jsonify(DataStore().list_documents())


# ---------------------------------------------------------------------------
# GET /api/files/<file_id>
# ---------------------------------------------------------------------------

@bp.route("/files/<file_id>", methods=["GET"])
def get_file(file_id: str):
    doc = DataStore().get_document(file_id)
    if doc is None:
        return jsonify({"error": f"Document '{file_id}' not found"}), 404

    return jsonify(
        {
            "file_id": doc.file_id,
            "filename": doc.filename,
            "uploaded_at": doc.uploaded_at,
            "extraction_method": doc.extraction_method,
            "chunk_count": len(doc.chunks),
            "indexed": doc.vector_store is not None,
            "extracted_data": doc.extracted_json,
        }
    )
