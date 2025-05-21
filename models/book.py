from sqlalchemy import Column, Integer, Float, REAL, VARCHAR, TEXT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Book(Base):
    __tablename__ = 'book'
    

    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR)
    rating_dist1 = Column(Integer)
    pages_number = Column(Integer)
    rating_dist4 = Column(Integer)
    rating_dist_total = Column(Integer)
    publisher = Column(VARCHAR)
    counts_of_review = Column(Integer)
    publish_year = Column(Integer)
    language = Column(VARCHAR)
    authors = Column(TEXT)
    rating_dist2 = Column(Integer)
    rating_dist5 = Column(Integer)
    isbn = Column(VARCHAR)
    rating_dist3 = Column(Integer)
    rating = Column(REAL)