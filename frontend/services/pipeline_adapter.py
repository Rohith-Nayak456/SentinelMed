"""
SentinelMed Backend Pipeline Adapter
Serves as the isolated integration boundary between the Streamlit UI and the
actual SentinelMed ML pipeline (P1-P5).

STATUS: PENDING BACKEND INTEGRATION
Do not invent backend module imports, function signatures, or mock success responses.
When called before teammates provide the pipeline interface, this adapter raises
BackendNotConfiguredError with explicit connection instructions.
"""

from typing import Any, Dict, Optional
from PIL import Image


class BackendNotConfiguredError(Exception):
    """Raised when live inference is attempted before teammate backend modules are wired."""
    pass


class PipelineAdapter:
    """
    Adapter boundary connecting the frontend to the SentinelMed reliability pipeline.
    
    Teammates: When the backend pipeline is ready, connect your entrypoint in
    `run_inference()` below and return a dictionary conforming to the SentinelMed
    result contract.
    """

    def __init__(self):
        # Once the actual backend is confirmed, initialize model clients,
        # pipelines, or queue consumers here.
        self._is_connected = False

    def is_configured(self) -> bool:
        """Indicates whether a live backend pipeline is wired and ready."""
        return self._is_connected

    def get_status_description(self) -> str:
        """Human-readable explanation of backend connection status."""
        if self._is_connected:
            return "Live SentinelMed backend pipeline connected."
        return (
            "Live backend pipeline is pending integration. Teammate modules (P1-P5) "
            "have not yet been connected to this adapter."
        )

    def run_inference(
        self,
        image: Image.Image,
        scan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the live SentinelMed reliability inspection pipeline.

        Args:
            image: Uploaded PIL Image.
            scan_id: Optional identifier for the scan.

        Returns:
            Dict conforming to the SentinelMed result contract.

        Raises:
            BackendNotConfiguredError: If live backend modules are not wired.
        """
        if not self._is_connected:
            raise BackendNotConfiguredError(
                "Live SentinelMed backend pipeline is not configured.\n\n"
                "To connect the real pipeline:\n"
                "1. Import your P1-P5 pipeline entry point in `frontend/services/pipeline_adapter.py`.\n"
                "2. Call your inspection, inference, gate, and recovery functions.\n"
                "3. Return the confirmed result dictionary.\n\n"
                "To explore the interface layout with synthetic example data, switch to 'Demo Mode' "
                "in the sidebar."
            )

        # ---------------------------------------------------------------------
        # TODO: TEAMMATE BACKEND INTEGRATION
        # Example once teammate modules are pushed:
        #
        # from backend.pipeline import run_sentinel_inspection
        # raw_result = run_sentinel_inspection(image, scan_id=scan_id)
        # return raw_result
        # ---------------------------------------------------------------------
        raise NotImplementedError("Live backend integration pending confirmation of P1-P5 modules.")
