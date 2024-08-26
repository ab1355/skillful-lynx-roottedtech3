from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128))
    is_admin = Column(Boolean, default=False)
    role_id = Column(Integer, ForeignKey('roles.id'))
    role = relationship('Role', back_populates='users')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    users = relationship('User', back_populates='role')

# Create database engine and session
engine = create_engine('sqlite:///agentzero.db')
Session = sessionmaker(bind=engine)
db_session = Session()

# Create tables
Base.metadata.create_all(engine)

# Initialize roles and admin user
def init_db():
    admin_role = db_session.query(Role).filter_by(name='Admin').first()
    if not admin_role:
        admin_role = Role(name='Admin')
        db_session.add(admin_role)

    hr_role = db_session.query(Role).filter_by(name='HR Manager').first()
    if not hr_role:
        hr_role = Role(name='HR Manager')
        db_session.add(hr_role)

    admin_user = db_session.query(User).filter_by(username='admin').first()
    if not admin_user:
        admin_user = User(username='admin', email='admin@example.com', is_admin=True, role=admin_role)
        admin_user.set_password('password123')
        db_session.add(admin_user)

    hr_user = db_session.query(User).filter_by(username='hr_manager').first()
    if not hr_user:
        hr_user = User(username='hr_manager', email='hr@example.com', role=hr_role)
        hr_user.set_password('hrpass456')
        db_session.add(hr_user)

    db_session.commit()

init_db()