import sys
import os

# Append src to sys path
sys.path.append(os.path.abspath('src'))

from sqlmodel import Session, select
from database.session import engine
from models.camera_network import CameraNetwork

with Session(engine) as session:
    networks = session.exec(select(CameraNetwork).where(CameraNetwork.status.in_(["running", "stopping"]))).all()
    for net in networks:
        print(f"Fixing network {net.id} from {net.status} to failed")
        net.status = "failed"
        session.add(net)
    session.commit()
    print("Done")
