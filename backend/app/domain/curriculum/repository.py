"""Hierarchical Curriculum Repository and Question Bank.

Defines a 4-tier Knowledge Component (KC) curriculum:
1. KC 0: Basic Arithmetic (Addition, Subtraction, Single-digit Multiplication, Basic Division)
2. KC 1: Advanced Arithmetic (Order of operations, Multi-digit multiplication, Fractions/Decimals)
3. KC 2: Basic Algebra (Single variable linear equations, expression evaluation)
4. KC 3: Advanced Algebra (Systems of equations, Quadratic expressions, Inequalities)

Each KC contains 21 calibrated questions (7 Easy, 7 Medium, 7 Hard), amounting to 84 total questions.
"""

from typing import Dict, List, Optional
from app.domain.curriculum.models import KnowledgeComponent, QuestionItem

# --- Knowledge Component Taxonomy & Prerequisite DAG ---
KNOWLEDGE_COMPONENTS: Dict[int, KnowledgeComponent] = {
    0: KnowledgeComponent(
        idx=0,
        name="Basic Arithmetic",
        description="Foundational arithmetic operations: addition, subtraction, multiplication, and simple division.",
        prerequisites=[],
    ),
    1: KnowledgeComponent(
        idx=1,
        name="Advanced Arithmetic",
        description="Multi-digit arithmetic, order of operations (PEMDAS), and fractions/decimals.",
        prerequisites=[0],
    ),
    2: KnowledgeComponent(
        idx=2,
        name="Basic Algebra",
        description="One-step and two-step linear equations, variable isolation, and substitution.",
        prerequisites=[0, 1],
    ),
    3: KnowledgeComponent(
        idx=3,
        name="Advanced Algebra",
        description="Quadratic equations, systems of linear equations, and multi-step algebraic manipulation.",
        prerequisites=[1, 2],
    ),
}

KC_NAMES: List[str] = [kc.name for kc in KNOWLEDGE_COMPONENTS.values()]

# --- Comprehensive Question Repository (84 Items) ---
QUESTION_BANK: Dict[int, List[QuestionItem]] = {
    # -------------------------------------------------------------
    # KC 0: BASIC ARITHMETIC (21 items: 7 Easy, 7 Medium, 7 Hard)
    # -------------------------------------------------------------
    0: [
        # Easy (0.2)
        QuestionItem("BA_E01", 0, "Basic Arithmetic", "Easy", 0.2, "What is 5 + 7?", 12.0, "5 + 7 = 12", "Count 7 units up from 5.", "Example: Add 4 + 3", "4 + 3 = 7"),
        QuestionItem("BA_E02", 0, "Basic Arithmetic", "Easy", 0.2, "What is 14 - 6?", 8.0, "14 - 6 = 8", "Subtract 6 from 14.", "Example: Subtract 10 - 4", "10 - 4 = 6"),
        QuestionItem("BA_E03", 0, "Basic Arithmetic", "Easy", 0.2, "What is 3 × 4?", 12.0, "3 × 4 = 12", "Multiply 3 groups of 4.", "Example: Multiply 2 × 5", "2 × 5 = 10"),
        QuestionItem("BA_E04", 0, "Basic Arithmetic", "Easy", 0.2, "What is 16 ÷ 4?", 4.0, "16 ÷ 4 = 4", "Find how many 4s go into 16.", "Example: Divide 12 ÷ 3", "12 ÷ 3 = 4"),
        QuestionItem("BA_E05", 0, "Basic Arithmetic", "Easy", 0.2, "What is 9 + 8?", 17.0, "9 + 8 = 17", "Combine 9 and 8.", "Example: Add 6 + 6", "6 + 6 = 12"),
        QuestionItem("BA_E06", 0, "Basic Arithmetic", "Easy", 0.2, "What is 15 - 7?", 8.0, "15 - 7 = 8", "Take away 7 from 15.", "Example: Subtract 13 - 5", "13 - 5 = 8"),
        QuestionItem("BA_E07", 0, "Basic Arithmetic", "Easy", 0.2, "What is 6 × 3?", 18.0, "6 × 3 = 18", "6 groups of 3.", "Example: Multiply 5 × 3", "5 × 3 = 15"),
        # Medium (0.5)
        QuestionItem("BA_M01", 0, "Basic Arithmetic", "Medium", 0.5, "What is 37 + 28?", 65.0, "37 + 28 = 65", "Add 37 + 20 = 57, then add 8 to get 65.", "Example: Add 45 + 19", "45 + 19 = 64"),
        QuestionItem("BA_M02", 0, "Basic Arithmetic", "Medium", 0.5, "What is 83 - 47?", 36.0, "83 - 47 = 36", "83 - 40 = 43, then 43 - 7 = 36.", "Example: Subtract 72 - 38", "72 - 38 = 34"),
        QuestionItem("BA_M03", 0, "Basic Arithmetic", "Medium", 0.5, "What is 7 × 8?", 56.0, "7 × 8 = 56", "Recall multiplication table 7 × 8.", "Example: Multiply 6 × 9", "6 × 9 = 54"),
        QuestionItem("BA_M04", 0, "Basic Arithmetic", "Medium", 0.5, "What is 72 ÷ 9?", 8.0, "72 ÷ 9 = 8", "9 times what equals 72? 9 × 8 = 72.", "Example: Divide 48 ÷ 6", "48 ÷ 6 = 8"),
        QuestionItem("BA_M05", 0, "Basic Arithmetic", "Medium", 0.5, "What is 46 + 39?", 85.0, "46 + 39 = 85", "46 + 40 = 86, then subtract 1.", "Example: Add 28 + 37", "28 + 37 = 65"),
        QuestionItem("BA_M06", 0, "Basic Arithmetic", "Medium", 0.5, "What is 94 - 58?", 36.0, "94 - 58 = 36", "94 - 50 = 44, then 44 - 8 = 36.", "Example: Subtract 61 - 25", "61 - 25 = 36"),
        QuestionItem("BA_M07", 0, "Basic Arithmetic", "Medium", 0.5, "What is 8 × 9?", 72.0, "8 × 9 = 72", "9 × 8 = 72.", "Example: Multiply 7 × 7", "7 × 7 = 49"),
        # Hard (0.8)
        QuestionItem("BA_H01", 0, "Basic Arithmetic", "Hard", 0.8, "What is 144 ÷ 12?", 12.0, "144 ÷ 12 = 12", "12 × 12 = 144.", "Example: Divide 121 ÷ 11", "121 ÷ 11 = 11"),
        QuestionItem("BA_H02", 0, "Basic Arithmetic", "Hard", 0.8, "What is 13 × 7?", 91.0, "13 × 7 = (10 × 7) + (3 × 7) = 70 + 21 = 91", "Break 13 into 10 and 3.", "Example: Multiply 14 × 6", "10 × 6 + 4 × 6 = 84"),
        QuestionItem("BA_H03", 0, "Basic Arithmetic", "Hard", 0.8, "What is 156 - 89?", 67.0, "156 - 89 = 156 - 90 + 1 = 66 + 1 = 67", "Subtract 90, then add back 1.", "Example: Subtract 142 - 79", "142 - 80 + 1 = 63"),
        QuestionItem("BA_H04", 0, "Basic Arithmetic", "Hard", 0.8, "What is 17 × 4?", 68.0, "17 × 4 = (10 × 4) + (7 × 4) = 40 + 28 = 68", "Multiply 10 × 4 and 7 × 4.", "Example: Multiply 16 × 5", "10 × 5 + 6 × 5 = 80"),
        QuestionItem("BA_H05", 0, "Basic Arithmetic", "Hard", 0.8, "What is 168 ÷ 14?", 12.0, "168 ÷ 14 = 12", "14 × 10 = 140; 168 - 140 = 28 = 14 × 2; total = 12.", "Example: Divide 156 ÷ 13", "13 × 12 = 156"),
        QuestionItem("BA_H06", 0, "Basic Arithmetic", "Hard", 0.8, "What is 23 × 6?", 138.0, "23 × 6 = (20 × 6) + (3 × 6) = 120 + 18 = 138", "Compute 20 × 6 + 3 × 6.", "Example: Multiply 24 × 5", "120"),
        QuestionItem("BA_H07", 0, "Basic Arithmetic", "Hard", 0.8, "What is 225 ÷ 15?", 15.0, "225 ÷ 15 = 15", "15 squared is 225.", "Example: Divide 196 ÷ 14", "196 ÷ 14 = 14"),
    ],
    # -----------------------------------------------------------------
    # KC 1: ADVANCED ARITHMETIC (21 items: 7 Easy, 7 Medium, 7 Hard)
    # -----------------------------------------------------------------
    1: [
        # Easy (0.2)
        QuestionItem("AA_E01", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 4 + 3 × 2", 10.0, "PEMDAS: 3 × 2 = 6, then 4 + 6 = 10.", "Multiply before adding.", "Example: 5 + 2 × 4", "5 + 8 = 13"),
        QuestionItem("AA_E02", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: (8 - 3) × 4", 20.0, "Parentheses first: 8 - 3 = 5; 5 × 4 = 20.", "Calculate inside parentheses first.", "Example: (7 - 2) × 3", "5 × 3 = 15"),
        QuestionItem("AA_E03", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 12 - 6 ÷ 2", 9.0, "Divide first: 6 ÷ 2 = 3; 12 - 3 = 9.", "Perform division before subtraction.", "Example: 10 - 8 ÷ 2", "10 - 4 = 6"),
        QuestionItem("AA_E04", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 5 × 3 - 7", 8.0, "Multiply first: 15 - 7 = 8.", "Multiply first, then subtract.", "Example: 4 × 4 - 6", "16 - 6 = 10"),
        QuestionItem("AA_E05", 1, "Advanced Arithmetic", "Easy", 0.2, "What is 0.5 + 0.25?", 0.75, "0.50 + 0.25 = 0.75.", "Think of decimals as cents: 50 cents + 25 cents.", "Example: 0.2 + 0.3", "0.5"),
        QuestionItem("AA_E06", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 18 ÷ (3 × 2)", 3.0, "Parentheses first: 3 × 2 = 6; 18 ÷ 6 = 3.", "Compute denominator in parentheses first.", "Example: 20 ÷ (2 × 5)", "20 ÷ 10 = 2"),
        QuestionItem("AA_E07", 1, "Advanced Arithmetic", "Easy", 0.2, "What is 2.5 × 2?", 5.0, "2.5 × 2 = 5.0.", "Doubling 2.5 gives 5.", "Example: 1.5 × 4", "6.0"),
        # Medium (0.5)
        QuestionItem("AA_M01", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 24 ÷ (3 + 1) × 2", 12.0, "Parentheses: 4; 24 ÷ 4 = 6; 6 × 2 = 12.", "Resolve (3+1) first, then proceed left to right.", "Example: 18 ÷ (2 + 1) × 2", "18 ÷ 3 × 2 = 12"),
        QuestionItem("AA_M02", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 3^2 + 4^2", 25.0, "3^2 = 9, 4^2 = 16, 9 + 16 = 25.", "Calculate exponents first.", "Example: 2^2 + 3^2", "4 + 9 = 13"),
        QuestionItem("AA_M03", 1, "Advanced Arithmetic", "Medium", 0.5, "What is 3/4 + 1/2 as a decimal?", 1.25, "3/4 = 0.75, 1/2 = 0.50, 0.75 + 0.50 = 1.25.", "Convert fractions to decimals or common denominator.", "Example: 1/4 + 1/2", "0.25 + 0.5 = 0.75"),
        QuestionItem("AA_M04", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 50 - 3 × (4 + 6)", 20.0, "Parentheses: 4 + 6 = 10; 3 × 10 = 30; 50 - 30 = 20.", "Work out parentheses, then multiply, then subtract.", "Example: 40 - 2 × (5 + 5)", "40 - 20 = 20"),
        QuestionItem("AA_M05", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 15% of 80", 12.0, "0.15 × 80 = 12.", "10% of 80 is 8, 5% is 4; 8 + 4 = 12.", "Example: 20% of 60", "0.20 × 60 = 12"),
        QuestionItem("AA_M06", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: (14 - 2^3) × 3", 18.0, "2^3 = 8; 14 - 8 = 6; 6 × 3 = 18.", "Compute power inside parentheses first.", "Example: (10 - 2^2) × 2", "6 × 2 = 12"),
        QuestionItem("AA_M07", 1, "Advanced Arithmetic", "Medium", 0.5, "What is 7.2 ÷ 0.8?", 9.0, "72 ÷ 8 = 9.", "Multiply numerator and denominator by 10.", "Example: 4.5 ÷ 0.5", "45 ÷ 5 = 9"),
        # Hard (0.8)
        QuestionItem("AA_H01", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: (5^2 - 3^2) ÷ (2^3 - 4)", 4.0, "(25 - 9) ÷ (8 - 4) = 16 ÷ 4 = 4.", "Evaluate exponents in numerator and denominator.", "Example: (4^2 - 2^2) ÷ 3", "12 ÷ 3 = 4"),
        QuestionItem("AA_H02", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: 12.5% of 320", 40.0, "12.5% = 1/8; 320 ÷ 8 = 40.", "Notice 12.5% is exactly 1/8.", "Example: 12.5% of 160", "160 / 8 = 20"),
        QuestionItem("AA_H03", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: 3/5 × 25/6", 2.5, "(3 × 25) / (5 × 6) = 75 / 30 = 2.5.", "Cross cancel 25 with 5 and 3 with 6.", "Example: 2/3 × 9/4", "18/12 = 1.5"),
        QuestionItem("AA_H04", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: sqrt(144) + 2^4 - 7", 21.0, "sqrt(144) = 12; 2^4 = 16; 12 + 16 - 7 = 21.", "Compute square root and power first.", "Example: sqrt(81) + 2^3", "9 + 8 = 17"),
        QuestionItem("AA_H05", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: 0.125 × 64", 8.0, "0.125 = 1/8; 64 ÷ 8 = 8.", "0.125 is 1/8.", "Example: 0.125 × 32", "4.0"),
        QuestionItem("AA_H06", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: 48 ÷ (2 × (9 - 6)^2)", 2.67, "48 ÷ (2 × 9) = 48 ÷ 18 = 2.67 (or 8/3)", "Compute parentheses (9-6)=3, square to get 9.", "Example: 36 ÷ (2 × 3)", "6.0"),
        QuestionItem("AA_H07", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: (7/8 - 1/4) as a decimal", 0.625, "7/8 - 2/8 = 5/8 = 0.625.", "Convert 1/4 to 2/8.", "Example: 3/4 - 1/2", "1/4 = 0.25"),
    ],
    # -------------------------------------------------------------
    # KC 2: BASIC ALGEBRA (21 items: 7 Easy, 7 Medium, 7 Hard)
    # -------------------------------------------------------------
    2: [
        # Easy (0.2)
        QuestionItem("BA_ALG_E01", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x + 7 = 15", 8.0, "x = 15 - 7 = 8.", "Subtract 7 from both sides.", "Example: x + 4 = 10", "x = 6"),
        QuestionItem("BA_ALG_E02", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: 3x = 21", 7.0, "x = 21 ÷ 3 = 7.", "Divide both sides by 3.", "Example: 4x = 24", "x = 6"),
        QuestionItem("BA_ALG_E03", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x - 9 = 14", 23.0, "x = 14 + 9 = 23.", "Add 9 to both sides.", "Example: x - 5 = 12", "x = 17"),
        QuestionItem("BA_ALG_E04", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x / 4 = 6", 24.0, "x = 6 × 4 = 24.", "Multiply both sides by 4.", "Example: x / 3 = 5", "x = 15"),
        QuestionItem("BA_ALG_E05", 2, "Basic Algebra", "Easy", 0.2, "If y = 3, evaluate: 2y + 5", 11.0, "2(3) + 5 = 6 + 5 = 11.", "Substitute 3 in place of y.", "Example: If y = 4, 3y + 2", "3(4) + 2 = 14"),
        QuestionItem("BA_ALG_E06", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: 2x = 18", 9.0, "x = 18 ÷ 2 = 9.", "Divide both sides by 2.", "Example: 5x = 35", "x = 7"),
        QuestionItem("BA_ALG_E07", 2, "Basic Algebra", "Easy", 0.2, "If a = 5 and b = 2, evaluate: 3a - 4b", 7.0, "3(5) - 4(2) = 15 - 8 = 7.", "Substitute a=5 and b=2.", "Example: 2a - b", "10 - 2 = 8"),
        # Medium (0.5)
        QuestionItem("BA_ALG_M01", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 2x + 5 = 19", 7.0, "2x = 14; x = 7.", "Subtract 5 first, then divide by 2.", "Example: 3x + 4 = 19", "3x = 15 => x = 5"),
        QuestionItem("BA_ALG_M02", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 5x - 8 = 27", 7.0, "5x = 35; x = 7.", "Add 8 to both sides, then divide by 5.", "Example: 4x - 6 = 26", "4x = 32 => x = 8"),
        QuestionItem("BA_ALG_M03", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 3(x + 4) = 30", 6.0, "x + 4 = 10; x = 6.", "Divide both sides by 3 or distribute.", "Example: 2(x + 5) = 24", "x + 5 = 12 => x = 7"),
        QuestionItem("BA_ALG_M04", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 4x + 3 = 2x + 15", 6.0, "2x = 12; x = 6.", "Subtract 2x and subtract 3 from both sides.", "Example: 5x + 2 = 3x + 10", "2x = 8 => x = 4"),
        QuestionItem("BA_ALG_M05", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: (x / 3) + 7 = 12", 15.0, "x / 3 = 5; x = 15.", "Subtract 7, then multiply by 3.", "Example: (x / 2) + 4 = 9", "x / 2 = 5 => x = 10"),
        QuestionItem("BA_ALG_M06", 2, "Basic Algebra", "Medium", 0.5, "If 4(y - 2) = 28, find y", 9.0, "y - 2 = 7; y = 9.", "Divide by 4 then add 2.", "Example: 3(y - 1) = 18", "y - 1 = 6 => y = 7"),
        QuestionItem("BA_ALG_M07", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 7 - 2x = -3", 5.0, "-2x = -10; x = 5.", "Subtract 7 from both sides: -2x = -10.", "Example: 8 - 3x = -7", "-3x = -15 => x = 5"),
        # Hard (0.8)
        QuestionItem("BA_ALG_H01", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 2(3x - 1) + 4 = 4x + 12", 5.0, "6x - 2 + 4 = 4x + 12 => 6x + 2 = 4x + 12 => 2x = 10 => x = 5.", "Expand 2(3x-1) first.", "Example: 3(2x - 1) = 4x + 7", "6x - 3 = 4x + 7 => 2x = 10 => x = 5"),
        QuestionItem("BA_ALG_H02", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: (2x + 6) / 4 = x - 1", 5.0, "2x + 6 = 4(x - 1) => 2x + 6 = 4x - 4 => 2x = 10 => x = 5.", "Multiply through by 4.", "Example: (3x + 3)/3 = x + 1", "Identity"),
        QuestionItem("BA_ALG_H03", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 5 - 2(x + 3) = -11", 5.0, "5 - 2x - 6 = -11 => -2x - 1 = -11 => -2x = -10 => x = 5.", "Distribute -2 carefully.", "Example: 4 - 3(x + 1) = -14", "-3x + 1 = -14 => x = 5"),
        QuestionItem("BA_ALG_H04", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: (3x - 5) / 2 = (x + 7) / 3", 4.14, "3(3x - 5) = 2(x + 7) => 9x - 15 = 2x + 14 => 7x = 29 => x = 4.14", "Cross multiply the denominators.", "Example: (x + 1)/2 = 3", "x = 5"),
        QuestionItem("BA_ALG_H05", 2, "Basic Algebra", "Hard", 0.8, "If 3x + 2y = 22 and y = 5, find x", 4.0, "3x + 10 = 22 => 3x = 12 => x = 4.", "Substitute y = 5 into the equation.", "Example: 2x + 3y = 17, y = 3", "2x = 8 => x = 4"),
        QuestionItem("BA_ALG_H06", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 4x - 3(2 - x) = 22", 4.0, "4x - 6 + 3x = 22 => 7x = 28 => x = 4.", "Distribute -3 into (2 - x).", "Example: 2x - (4 - x) = 8", "3x = 12 => x = 4"),
        QuestionItem("BA_ALG_H07", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 0.4x + 1.2 = 2.8", 4.0, "0.4x = 1.6 => x = 4.", "Subtract 1.2, then divide by 0.4.", "Example: 0.5x + 1 = 3", "x = 4"),
    ],
    # -----------------------------------------------------------------
    # KC 3: ADVANCED ALGEBRA (21 items: 7 Easy, 7 Medium, 7 Hard)
    # -----------------------------------------------------------------
    3: [
        # Easy (0.2)
        QuestionItem("AA_ALG_E01", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: x^2 = 36", 6.0, "x = sqrt(36) = 6.", "Take the positive square root of 36.", "Example: x^2 = 25", "x = 5"),
        QuestionItem("AA_ALG_E02", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: x^2 - 49 = 0", 7.0, "x^2 = 49 => x = 7.", "Add 49 to both sides.", "Example: x^2 - 16 = 0", "x = 4"),
        QuestionItem("AA_ALG_E03", 3, "Advanced Algebra", "Easy", 0.2, "If x + y = 10 and x - y = 4, find x", 7.0, "Add equations: 2x = 14 => x = 7.", "Add both equations together to eliminate y.", "Example: x + y = 8, x - y = 2", "2x = 10 => x = 5"),
        QuestionItem("AA_ALG_E04", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: 2x^2 = 50", 5.0, "x^2 = 25 => x = 5.", "Divide by 2 first, then take the square root.", "Example: 3x^2 = 27", "x^2 = 9 => x = 3"),
        QuestionItem("AA_ALG_E05", 3, "Advanced Algebra", "Easy", 0.2, "Find the positive root: (x - 3)(x + 5) = 0", 3.0, "Roots are 3 and -5. Positive root is 3.", "Set each factor equal to zero.", "Example: (x - 4)(x + 2) = 0", "x = 4"),
        QuestionItem("AA_ALG_E06", 3, "Advanced Algebra", "Easy", 0.2, "If x + y = 12 and y = 4, find x", 8.0, "x + 4 = 12 => x = 8.", "Substitute y=4.", "Example: x + y = 15, y = 5", "x = 10"),
        QuestionItem("AA_ALG_E07", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: x^2 + 5 = 30", 5.0, "x^2 = 25 => x = 5.", "Subtract 5, then take square root.", "Example: x^2 + 4 = 20", "x = 4"),
        # Medium (0.5)
        QuestionItem("AA_ALG_M01", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive root: x^2 - 5x + 6 = 0 (larger root)", 3.0, "(x - 2)(x - 3) = 0 => roots 2 and 3. Larger is 3.", "Factor into (x - a)(x - b) = 0.", "Example: x^2 - 3x + 2 = 0", "Roots 1, 2 => larger is 2"),
        QuestionItem("AA_ALG_M02", 3, "Advanced Algebra", "Medium", 0.5, "System: 2x + y = 11, x - y = 1. Find x.", 4.0, "Add equations: 3x = 12 => x = 4.", "Add the equations to cancel y.", "Example: 3x + y = 14, x - y = 2", "4x = 16 => x = 4"),
        QuestionItem("AA_ALG_M03", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive x: (x + 2)^2 = 49", 5.0, "x + 2 = 7 => x = 5.", "Take square root of both sides: x + 2 = 7.", "Example: (x + 1)^2 = 36", "x + 1 = 6 => x = 5"),
        QuestionItem("AA_ALG_M04", 3, "Advanced Algebra", "Medium", 0.5, "Find vertex x-coordinate for f(x) = x^2 - 6x + 8", 3.0, "x_v = -b / (2a) = 6 / 2 = 3.", "Use vertex formula x = -b / (2a).", "Example: f(x) = x^2 - 4x + 1", "x = 4 / 2 = 2"),
        QuestionItem("AA_ALG_M05", 3, "Advanced Algebra", "Medium", 0.5, "System: 3x + 2y = 16, y = 2. Find x.", 4.0, "3x + 4 = 16 => 3x = 12 => x = 4.", "Substitute y=2.", "Example: 2x + 4y = 20, y = 3", "2x = 8 => x = 4"),
        QuestionItem("AA_ALG_M06", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive root: x^2 - 7x + 10 = 0 (larger root)", 5.0, "(x - 2)(x - 5) = 0 => x = 5.", "Factor 10 into 2 and 5.", "Example: x^2 - 6x + 8 = 0", "Roots 2, 4 => 4"),
        QuestionItem("AA_ALG_M07", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive x: 3x^2 - 12 = 0", 2.0, "3x^2 = 12 => x^2 = 4 => x = 2.", "Divide by 3, take square root.", "Example: 2x^2 - 18 = 0", "x = 3"),
        # Hard (0.8)
        QuestionItem("AA_ALG_H01", 3, "Advanced Algebra", "Hard", 0.8, "System: 3x + 4y = 26, 2x - y = 2. Find x.", 3.1, "Multiply 2nd by 4: 8x - 4y = 8. Add: 11x = 34 => x = 3.09", "Use elimination by multiplying second equation.", "Example: 2x + 3y = 12, x - y = 1", "x = 3"),
        QuestionItem("AA_ALG_H02", 3, "Advanced Algebra", "Hard", 0.8, "Find positive root of 2x^2 - 7x + 3 = 0 (larger root)", 3.0, "(2x - 1)(x - 3) = 0 => x = 3.", "Factor using ac method: 2 × 3 = 6.", "Example: 2x^2 - 5x + 2 = 0", "x = 2"),
        QuestionItem("AA_ALG_H03", 3, "Advanced Algebra", "Hard", 0.8, "System: 4x - 3y = 5, 2x + y = 5. Find x.", 2.0, "Multiply 2nd by 3: 6x + 3y = 15. Add: 10x = 20 => x = 2.", "Multiply 2nd equation by 3 and add.", "Example: 3x - y = 5, x + y = 3", "4x = 8 => x = 2"),
        QuestionItem("AA_ALG_H04", 3, "Advanced Algebra", "Hard", 0.8, "Solve for positive x: x^2 - 8x + 12 = 0 (larger root)", 6.0, "(x - 2)(x - 6) = 0 => x = 6.", "Find two numbers that multiply to 12 and add to -8.", "Example: x^2 - 9x + 14 = 0", "x = 7"),
        QuestionItem("AA_ALG_H05", 3, "Advanced Algebra", "Hard", 0.8, "Find discriminant of quadratic 2x^2 - 4x + 2 = 0", 0.0, "b^2 - 4ac = (-4)^2 - 4(2)(2) = 16 - 16 = 0.", "Use formula Delta = b^2 - 4ac.", "Example: x^2 - 4x + 4 = 0", "16 - 16 = 0"),
        QuestionItem("AA_ALG_H06", 3, "Advanced Algebra", "Hard", 0.8, "System: 5x + 2y = 29, 3x - y = 13. Find x.", 5.0, "Multiply 2nd by 2: 6x - 2y = 26. Add: 11x = 55 => x = 5.", "Eliminate y by multiplying by 2.", "Example: 4x + y = 14, 2x - y = 4", "6x = 18 => x = 3"),
        QuestionItem("AA_ALG_H07", 3, "Advanced Algebra", "Hard", 0.8, "Find positive x: 2(x - 1)^2 = 32", 5.0, "(x - 1)^2 = 16 => x - 1 = 4 => x = 5.", "Divide by 2, take square root, add 1.", "Example: 3(x - 2)^2 = 27", "x - 2 = 3 => x = 5"),
    ],
}


def get_all_kcs() -> List[KnowledgeComponent]:
    """Returns all Knowledge Components."""
    return list(KNOWLEDGE_COMPONENTS.values())


def get_kc_by_idx(kc_idx: int) -> Optional[KnowledgeComponent]:
    """Fetches KC definition by index."""
    return KNOWLEDGE_COMPONENTS.get(kc_idx)


def get_questions_by_kc(
    kc_idx: int, difficulty: Optional[str] = None
) -> List[QuestionItem]:
    """Retrieves all questions for a given KC, optionally filtered by difficulty."""
    questions = QUESTION_BANK.get(kc_idx, [])
    if difficulty:
        return [q for q in questions if q.difficulty.lower() == difficulty.lower()]
    return list(questions)


def get_question_by_id(question_id: str) -> Optional[QuestionItem]:
    """Finds a question item across all Knowledge Components by its ID."""
    for items in QUESTION_BANK.values():
        for item in items:
            if item.id == question_id:
                return item
    return None


def get_prerequisites(kc_idx: int) -> List[int]:
    """Returns prerequisite KC indices for the given KC index."""
    kc = KNOWLEDGE_COMPONENTS.get(kc_idx)
    return kc.prerequisites if kc else []


def check_prerequisites_met(kc_idx: int, beliefs: List[float], threshold: float = 0.70) -> bool:
    """Checks whether all prerequisites for a KC have reached the mastery threshold."""
    prereqs = get_prerequisites(kc_idx)
    for p_idx in prereqs:
        if p_idx < len(beliefs) and beliefs[p_idx] < threshold:
            return False
    return True
