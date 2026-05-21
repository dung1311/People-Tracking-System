1. **Model & Database Migration**:
   - Rename `TrackingSession` model to `CameraNetwork`, table `cameranetwork`.
   - In `Camera`, add `network_id` (foreign key to `cameranetwork.id`).
   - Create an Alembic migration to:
     - Rename table `trackingsession` to `cameranetwork`.
     - Add `network_id` column to `camera`.
     - Remove `camera_ids` JSON column from `cameranetwork`.
2. **Backend API**:
   - Rename `sessions.py` to `camera_networks.py`.
   - Update `POST /api/v1/camera-networks` to create empty network.
   - Update `POST /api/v1/cameras` or add `POST /api/v1/camera-networks/{id}/cameras` to add a camera directly to a network.
   - Update `core/session_manager.py` to stream from `CameraNetwork.cameras` instead of `camera_ids`.
3. **Frontend**:
   - Replace "Phiên Theo Dõi" / "Session" text with "Camera Network".
   - Modify `Sessions.tsx` -> `CameraNetworks.tsx`:
     - Create Camera Network only requires name/configs (no camera selection array).
   - In `SessionDetail.tsx` (now `CameraNetworkDetail.tsx`):
     - Show list of cameras in this network.
     - Add button "Add Camera" which opens the "Add Camera" modal from `Cameras.tsx`.
     - When submitting "Add Camera", it sets the `network_id` for that camera.
     - Keep "Start Pipeline" as is.
