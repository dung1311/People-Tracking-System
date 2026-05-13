"""create_gallery_track_table

Revision ID: c7d2f45f8a11
Revises: 9a6374a9a557
Create Date: 2026-04-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7d2f45f8a11'
down_revision: Union[str, Sequence[str], None] = '9a6374a9a557'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'gallery_track',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('person_id', sa.Integer(), nullable=False),
        sa.Column('tracker_id', sa.Integer(), nullable=False),
        sa.Column('is_mapped', sa.Boolean(), nullable=False),
        sa.Column('cam_id', sa.Integer(), nullable=False),
        sa.Column('frame_id', sa.Integer(), nullable=False),
        sa.Column('bbox', sa.JSON(), nullable=True),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('class_id', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('lost_age', sa.Integer(), nullable=False),
        sa.Column('hits', sa.Integer(), nullable=False),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_gallery_track_person_id'), 'gallery_track', ['person_id'], unique=True)
    op.create_index(op.f('ix_gallery_track_tracker_id'), 'gallery_track', ['tracker_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_gallery_track_tracker_id'), table_name='gallery_track')
    op.drop_index(op.f('ix_gallery_track_person_id'), table_name='gallery_track')
    op.drop_table('gallery_track')
