"""Authentication & User Identity API Endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import StudentProfile, User
from app.schemas.auth import (
    StudentProfileResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """Registers a new student or researcher account."""
    existing_user = db.query(User).filter(
        (User.username == request.username) | (User.email == request.email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )

    user = User(
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password),
        role=request.role or "student",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Automatically create student profile
    profile = StudentProfile(
        user_id=user.id,
        current_streak=0,
        total_sessions=0,
        total_questions_solved=0,
        mean_mastery=0.20,
        kc_0_belief=0.20,
        kc_1_belief=0.20,
        kc_2_belief=0.20,
        kc_3_belief=0.20,
    )
    db.add(profile)
    db.commit()

    return user


@router.post("/login", response_model=TokenResponse)
def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticates with username/password and returns signed JWT access token."""
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_access_token(subject=user.id, role=user.role)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        username=user.username,
        user_id=user.id,
    )


@router.post("/token", response_model=TokenResponse)
def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """Standard OAuth2 password grant for Swagger UI interactive testing."""
    return login(UserLoginRequest(username=form_data.username, password=form_data.password), db)


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Retrieves authenticated user profile and longitudinal mastery."""
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    profile_dto = None
    if profile:
        profile_dto = StudentProfileResponse(
            current_streak=profile.current_streak,
            total_sessions=profile.total_sessions,
            total_questions_solved=profile.total_questions_solved,
            mean_mastery=profile.mean_mastery,
            kc_0_belief=profile.kc_0_belief,
            kc_1_belief=profile.kc_1_belief,
            kc_2_belief=profile.kc_2_belief,
            kc_3_belief=profile.kc_3_belief,
        )

    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        profile=profile_dto,
        created_at=current_user.created_at,
    )
