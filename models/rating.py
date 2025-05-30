from sqlalchemy import Column, Integer, Float, REAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Rating(Base):
    __tablename__ = 'rating'
    
    key = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    book_id = Column(Integer)
    rating = Column(REAL)