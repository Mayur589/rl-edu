"""Hierarchical Curriculum and Question Bank for Intelligent Tutoring System.

This module defines a 4-tier Knowledge Component (KC) curriculum covering:
1. Basic Arithmetic
2. Advanced Arithmetic
3. Basic Algebra
4. Advanced Algebra

Each Knowledge Component contains 20+ structured questions categorized by difficulty
(Easy, Medium, Hard), complete with worked examples and scaffolding hints for every question.
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class QuestionItem:
    """Represents an individual curriculum question item.

    Attributes:
        id (str): Unique identifier string.
        kc_idx (int): Knowledge Component index in {0, 1, 2, 3}.
        kc_name (str): Human-readable Knowledge Component title.
        difficulty (str): Difficulty classification ('Easy', 'Medium', 'Hard').
        difficulty_val (float): Numerical difficulty value in {0.2, 0.5, 0.8}.
        prompt (str): Mathematical question prompt text.
        answer (float): Numerical answer expected from student.
        explanation (str): Comprehensive step-by-step solution explanation.
        hint (str): Scaffolding hint providing immediate guidance.
        worked_example_prompt (str): Demonstration prompt for worked example mode.
        worked_example_explanation (str): Demonstration explanation for worked example mode.
    """

    id: str
    kc_idx: int
    kc_name: str
    difficulty: str
    difficulty_val: float
    prompt: str
    answer: float
    explanation: str
    hint: str
    worked_example_prompt: str
    worked_example_explanation: str


# Knowledge Component Taxonomy
KC_NAMES: List[str] = [
    "Basic Arithmetic",
    "Advanced Arithmetic",
    "Basic Algebra",
    "Advanced Algebra",
]

# --- Comprehensive Question Repository (21 items per KC = 84 total items) ---
QUESTION_BANK: Dict[int, List[QuestionItem]] = {
    # =========================================================================
    # KC 0: BASIC ARITHMETIC (Simple +, -, *, /)
    # =========================================================================
    0: [
        # Easy (0.2)
        QuestionItem("B0_E1", 0, "Basic Arithmetic", "Easy", 0.2, "What is 5 + 7?", 12.0, "5 + 7 = 12", "Combine 5 and 7 together.", "Example: Add 4 + 3", "4 + 3 = 7"),
        QuestionItem("B0_E2", 0, "Basic Arithmetic", "Easy", 0.2, "What is 14 - 6?", 8.0, "14 - 6 = 8", "Subtract 6 from 14.", "Example: Subtract 10 - 4", "10 - 4 = 6"),
        QuestionItem("B0_E3", 0, "Basic Arithmetic", "Easy", 0.2, "What is 3 × 4?", 12.0, "3 × 4 = 12", "Multiply 3 by 4.", "Example: Multiply 2 × 5", "2 × 5 = 10"),
        QuestionItem("B0_E4", 0, "Basic Arithmetic", "Easy", 0.2, "What is 12 ÷ 4?", 3.0, "12 ÷ 4 = 3", "Find how many 4s are in 12.", "Example: Divide 10 ÷ 2", "10 ÷ 2 = 5"),
        QuestionItem("B0_E5", 0, "Basic Arithmetic", "Easy", 0.2, "What is 9 + 8?", 17.0, "9 + 8 = 17", "Combine 9 and 8.", "Example: Add 6 + 6", "6 + 6 = 12"),
        QuestionItem("B0_E6", 0, "Basic Arithmetic", "Easy", 0.2, "What is 18 - 9?", 9.0, "18 - 9 = 9", "Subtract 9 from 18.", "Example: Subtract 15 - 7", "15 - 7 = 8"),
        QuestionItem("B0_E7", 0, "Basic Arithmetic", "Easy", 0.2, "What is 4 × 5?", 20.0, "4 × 5 = 20", "Multiply 4 by 5.", "Example: Multiply 3 × 3", "3 × 3 = 9"),
        # Medium (0.5)
        QuestionItem("B0_M1", 0, "Basic Arithmetic", "Medium", 0.5, "What is 37 + 28?", 65.0, "37 + 28 = 65", "Add 37 + 20 = 57, then add 8.", "Example: Add 45 + 19", "45 + 19 = 64"),
        QuestionItem("B0_M2", 0, "Basic Arithmetic", "Medium", 0.5, "What is 83 - 47?", 36.0, "83 - 47 = 36", "Subtract 40 from 83 = 43, then subtract 7.", "Example: Subtract 72 - 38", "72 - 38 = 34"),
        QuestionItem("B0_M3", 0, "Basic Arithmetic", "Medium", 0.5, "What is 7 × 8?", 56.0, "7 × 8 = 56", "Recall multiplication table 7 × 8.", "Example: Multiply 6 × 9", "6 × 9 = 54"),
        QuestionItem("B0_M4", 0, "Basic Arithmetic", "Medium", 0.5, "What is 72 ÷ 9?", 8.0, "72 ÷ 9 = 8", "9 × 8 = 72.", "Example: Divide 48 ÷ 6", "48 ÷ 6 = 8"),
        QuestionItem("B0_M5", 0, "Basic Arithmetic", "Medium", 0.5, "What is 46 + 39?", 85.0, "46 + 39 = 85", "46 + 40 = 86, subtract 1.", "Example: Add 28 + 37", "28 + 37 = 65"),
        QuestionItem("B0_M6", 0, "Basic Arithmetic", "Medium", 0.5, "What is 94 - 58?", 36.0, "94 - 58 = 36", "94 - 50 = 44, subtract 8.", "Example: Subtract 61 - 25", "61 - 25 = 36"),
        QuestionItem("B0_M7", 0, "Basic Arithmetic", "Medium", 0.5, "What is 8 × 9?", 72.0, "8 × 9 = 72", "9 × 8 = 72.", "Example: Multiply 7 × 7", "7 × 7 = 49"),
        # Hard (0.8)
        QuestionItem("B0_H1", 0, "Basic Arithmetic", "Hard", 0.8, "What is 144 ÷ 12?", 12.0, "144 ÷ 12 = 12", "12 × 12 = 144.", "Example: Divide 121 ÷ 11", "121 ÷ 11 = 11"),
        QuestionItem("B0_H2", 0, "Basic Arithmetic", "Hard", 0.8, "What is 15 × 12?", 180.0, "15 × 10 = 150, 15 × 2 = 30 ➔ 180", "Break 12 into (10 + 2).", "Example: Multiply 14 × 11", "14 × 11 = 154"),
        QuestionItem("B0_H3", 0, "Basic Arithmetic", "Hard", 0.8, "What is 342 - 187?", 155.0, "342 - 187 = 155", "342 - 200 = 142 + 13 = 155.", "Example: Subtract 415 - 268", "415 - 268 = 147"),
        QuestionItem("B0_H4", 0, "Basic Arithmetic", "Hard", 0.8, "What is 216 ÷ 18?", 12.0, "216 ÷ 18 = 12", "18 × 10 = 180, 18 × 2 = 36.", "Example: Divide 195 ÷ 15", "195 ÷ 15 = 13"),
        QuestionItem("B0_H5", 0, "Basic Arithmetic", "Hard", 0.8, "What is 27 × 15?", 405.0, "27 × 10 = 270, 27 × 5 = 135 ➔ 405", "Multiply 27 by 10 and add half of that.", "Example: Multiply 24 × 15", "24 × 15 = 360"),
        QuestionItem("B0_H6", 0, "Basic Arithmetic", "Hard", 0.8, "What is 512 - 279?", 233.0, "512 - 279 = 233", "512 - 300 = 212 + 21.", "Example: Subtract 604 - 388", "604 - 388 = 216"),
        QuestionItem("B0_H7", 0, "Basic Arithmetic", "Hard", 0.8, "What is 225 ÷ 15?", 15.0, "225 ÷ 15 = 15", "15 × 15 = 225.", "Example: Divide 169 ÷ 13", "169 ÷ 13 = 13"),
    ],

    # =========================================================================
    # KC 1: ADVANCED ARITHMETIC (PEMDAS, Fractions, Multi-step)
    # =========================================================================
    1: [
        # Easy (0.2)
        QuestionItem("A1_E1", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 3 * (4 + 5)", 27.0, "Parentheses first: 4 + 5 = 9. 3 * 9 = 27", "Solve parentheses (4 + 5) first.", "Example: Evaluate 2 * (3 + 4)", "3 + 4 = 7, 2 * 7 = 14"),
        QuestionItem("A1_E2", 1, "Advanced Arithmetic", "Easy", 0.2, "What is 1/2 + 1/4 in decimal form?", 0.75, "1/2 = 0.5, 1/4 = 0.25 ➔ 0.75", "Convert to decimals or common denominator 2/4 + 1/4 = 3/4.", "Example: 1/4 + 1/4", "2/4 = 0.5"),
        QuestionItem("A1_E3", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 10 - 2 * 3", 4.0, "Multiplication before subtraction: 2 * 3 = 6. 10 - 6 = 4", "Do multiplication before subtraction.", "Example: Evaluate 15 - 3 * 4", "3 * 4 = 12, 15 - 12 = 3"),
        QuestionItem("A1_E4", 1, "Advanced Arithmetic", "Easy", 0.2, "What is 3/4 as a decimal?", 0.75, "3 ÷ 4 = 0.75", "Divide 3 by 4.", "Example: 1/4 as decimal", "1 ÷ 4 = 0.25"),
        QuestionItem("A1_E5", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: 5 + 4 * 2", 13.0, "Multiply first: 4 * 2 = 8. 5 + 8 = 13", "PEMDAS: Multiplication before addition.", "Example: Evaluate 6 + 3 * 5", "3 * 5 = 15, 6 + 15 = 21"),
        QuestionItem("A1_E6", 1, "Advanced Arithmetic", "Easy", 0.2, "What is 2/5 as a decimal?", 0.4, "2 ÷ 5 = 0.4", "Divide 2 by 5.", "Example: 1/5 as decimal", "1 ÷ 5 = 0.2"),
        QuestionItem("A1_E7", 1, "Advanced Arithmetic", "Easy", 0.2, "Evaluate: (12 - 4) ÷ 2", 4.0, "12 - 4 = 8. 8 ÷ 2 = 4", "Solve parentheses first.", "Example: Evaluate (15 - 5) ÷ 5", "10 ÷ 5 = 2"),
        # Medium (0.5)
        QuestionItem("A1_M1", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 4 * 5 + 18 ÷ 3", 26.0, "4 * 5 = 20, 18 ÷ 3 = 6 ➔ 20 + 6 = 26", "Do multiplication and division before addition.", "Example: Evaluate 3 * 6 + 20 ÷ 4", "18 + 5 = 23"),
        QuestionItem("A1_M2", 1, "Advanced Arithmetic", "Medium", 0.5, "What is 3/4 + 5/8 in decimal form?", 1.375, "3/4 = 6/8. 6/8 + 5/8 = 11/8 = 1.375", "Find common denominator 8: 6/8 + 5/8 = 11/8.", "Example: 1/2 + 3/8", "4/8 + 3/8 = 7/8 = 0.875"),
        QuestionItem("A1_M3", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: (6 + 2)^2 - 15", 49.0, "6 + 2 = 8. 8^2 = 64. 64 - 15 = 49", "Parentheses first (8), square it (64), subtract 15.", "Example: Evaluate (5 + 1)^2 - 10", "6^2 = 36, 36 - 10 = 26"),
        QuestionItem("A1_M4", 1, "Advanced Arithmetic", "Medium", 0.5, "What is 7/10 - 2/5 as a decimal?", 0.3, "2/5 = 4/10. 7/10 - 4/10 = 3/10 = 0.3", "Convert 2/5 to 4/10.", "Example: 9/10 - 1/2", "9/10 - 5/10 = 0.4"),
        QuestionItem("A1_M5", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 50 - 4 * (3 + 2)^2", -50.0, "3 + 2 = 5, 5^2 = 25. 4 * 25 = 100. 50 - 100 = -50", "PEMDAS: Parentheses (5), Exponents (25), Multiply (100).", "Example: Evaluate 30 - 2 * (4 + 1)^2", "30 - 2 * 25 = -20"),
        QuestionItem("A1_M6", 1, "Advanced Arithmetic", "Medium", 0.5, "What is 5/6 * 3/4 as a decimal?", 0.625, "(5 * 3)/(6 * 4) = 15/24 = 5/8 = 0.625", "Multiply numerators and denominators: 15/24.", "Example: 2/3 * 3/4", "6/12 = 0.5"),
        QuestionItem("A1_M7", 1, "Advanced Arithmetic", "Medium", 0.5, "Evaluate: 18 ÷ (9 - 3) + 7 * 2", 17.0, "9 - 3 = 6. 18 ÷ 6 = 3. 7 * 2 = 14 ➔ 3 + 14 = 17", "Parentheses first (6), then division and multiplication.", "Example: Evaluate 24 ÷ (8 - 2) + 5 * 3", "24 ÷ 6 = 4, 4 + 15 = 19"),
        # Hard (0.8)
        QuestionItem("A1_H1", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: [24 - (3 + 1)^2] * 3", 24.0, "3 + 1 = 4. 4^2 = 16. 24 - 16 = 8. 8 * 3 = 24", "Inner parentheses first (4), square it (16), subtract from 24.", "Example: Evaluate [30 - (2 + 3)^2] * 4", "30 - 25 = 5, 5 * 4 = 20"),
        QuestionItem("A1_H2", 1, "Advanced Arithmetic", "Hard", 0.8, "What is 7/8 ÷ 3/4 in decimal form?", 1.1667, "7/8 * 4/3 = 28/24 = 7/6 ≈ 1.167", "Multiply by reciprocal: 7/8 * 4/3.", "Example: 3/4 ÷ 1/2", "3/4 * 2/1 = 6/4 = 1.5"),
        QuestionItem("A1_H3", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: 3 * 2^3 - 4 * (6 - 2)", 8.0, "2^3 = 8. 3 * 8 = 24. 6 - 2 = 4. 4 * 4 = 16. 24 - 16 = 8", "Exponents first (8), then multiplication.", "Example: Evaluate 2 * 3^2 - 3 * (5 - 1)", "18 - 12 = 6"),
        QuestionItem("A1_H4", 1, "Advanced Arithmetic", "Hard", 0.8, "What is (3/5 + 1/2) * 10?", 11.0, "3/5 = 0.6, 1/2 = 0.5 ➔ 1.1 * 10 = 11", "Add fractions: 6/10 + 5/10 = 11/10.", "Example: (1/4 + 1/2) * 8", "3/4 * 8 = 6"),
        QuestionItem("A1_H5", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: (15 - 3)^2 ÷ (4 * 3) + 7", 19.0, "12^2 = 144. 4 * 3 = 12. 144 ÷ 12 = 12. 12 + 7 = 19", "Solve parentheses, square 12, divide by 12.", "Example: Evaluate (10 - 2)^2 ÷ (2 * 4) + 5", "64 ÷ 8 + 5 = 13"),
        QuestionItem("A1_H6", 1, "Advanced Arithmetic", "Hard", 0.8, "What is 11/12 - 3/4 in decimal form?", 0.1667, "3/4 = 9/12. 11/12 - 9/12 = 2/12 = 1/6 ≈ 0.1667", "Common denominator 12: 11/12 - 9/12 = 2/12.", "Example: 5/6 - 1/2", "5/6 - 3/6 = 2/6 = 0.3333"),
        QuestionItem("A1_H7", 1, "Advanced Arithmetic", "Hard", 0.8, "Evaluate: 2^4 * 3 - 5 * (8 - 3)", 23.0, "2^4 = 16. 16 * 3 = 48. 8 - 3 = 5. 5 * 5 = 25. 48 - 25 = 23", "Exponents (16), multiply (48), subtract 25.", "Example: Evaluate 3^3 * 2 - 4 * (10 - 4)", "54 - 24 = 30"),
    ],

    # =========================================================================
    # KC 2: BASIC ALGEBRA (One-step & Two-step Linear Equations)
    # =========================================================================
    2: [
        # Easy (0.2)
        QuestionItem("L2_E1", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x + 5 = 12", 7.0, "Subtract 5 from both sides: x = 12 - 5 = 7", "Isolate x by subtracting 5.", "Example: Solve x + 4 = 10", "x = 10 - 4 = 6"),
        QuestionItem("L2_E2", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x - 8 = 15", 23.0, "Add 8 to both sides: x = 15 + 8 = 23", "Isolate x by adding 8.", "Example: Solve x - 3 = 9", "x = 9 + 3 = 12"),
        QuestionItem("L2_E3", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: 4x = 28", 7.0, "Divide both sides by 4: x = 28 ÷ 4 = 7", "Divide by coefficient 4.", "Example: Solve 3x = 18", "x = 18 ÷ 3 = 6"),
        QuestionItem("L2_E4", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x ÷ 3 = 6", 18.0, "Multiply both sides by 3: x = 6 * 3 = 18", "Multiply both sides by 3.", "Example: Solve x ÷ 5 = 4", "x = 4 * 5 = 20"),
        QuestionItem("L2_E5", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x + 9 = 21", 12.0, "Subtract 9 from both sides: x = 21 - 9 = 12", "Subtract 9 from 21.", "Example: Solve x + 7 = 15", "x = 15 - 7 = 8"),
        QuestionItem("L2_E6", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: x - 12 = 14", 26.0, "Add 12 to both sides: x = 14 + 12 = 26", "Add 12 to 14.", "Example: Solve x - 5 = 11", "x = 11 + 5 = 16"),
        QuestionItem("L2_E7", 2, "Basic Algebra", "Easy", 0.2, "Solve for x: 6x = 54", 9.0, "Divide by 6: x = 54 ÷ 6 = 9", "Divide 54 by 6.", "Example: Solve 5x = 35", "x = 35 ÷ 5 = 7"),
        # Medium (0.5)
        QuestionItem("L2_M1", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 2x - 4 = 10", 7.0, "Add 4: 2x = 14. Divide by 2: x = 7", "First add 4 to both sides, then divide by 2.", "Example: Solve 3x - 5 = 10", "3x = 15 ➔ x = 5"),
        QuestionItem("L2_M2", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 3x + 7 = 22", 5.0, "Subtract 7: 3x = 15. Divide by 3: x = 5", "Subtract 7 first, then divide by 3.", "Example: Solve 4x + 6 = 26", "4x = 20 ➔ x = 5"),
        QuestionItem("L2_M3", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 5x - 8 = 27", 7.0, "Add 8: 5x = 35. Divide by 5: x = 7", "Add 8 to 27.", "Example: Solve 2x - 9 = 11", "2x = 20 ➔ x = 10"),
        QuestionItem("L2_M4", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 4x + 12 = 40", 7.0, "Subtract 12: 4x = 28. Divide by 4: x = 7", "Subtract 12, then divide by 4.", "Example: Solve 6x + 8 = 38", "6x = 30 ➔ x = 5"),
        QuestionItem("L2_M5", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: (x + 3) / 2 = 8", 13.0, "Multiply by 2: x + 3 = 16. Subtract 3: x = 13", "Multiply both sides by 2 first.", "Example: Solve (x + 2) / 3 = 5", "x + 2 = 15 ➔ x = 13"),
        QuestionItem("L2_M6", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 7x - 15 = 34", 7.0, "Add 15: 7x = 49. Divide by 7: x = 7", "Add 15 to 34.", "Example: Solve 8x - 12 = 44", "8x = 56 ➔ x = 7"),
        QuestionItem("L2_M7", 2, "Basic Algebra", "Medium", 0.5, "Solve for x: 9x + 4 = 40", 4.0, "Subtract 4: 9x = 36. Divide by 9: x = 4", "Subtract 4, then divide by 9.", "Example: Solve 7x + 5 = 26", "7x = 21 ➔ x = 3"),
        # Hard (0.8)
        QuestionItem("L2_H1", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 5x - 3 = 2x + 12", 5.0, "Subtract 2x: 3x - 3 = 12. Add 3: 3x = 15. Divide: x = 5", "Gather x terms on one side: 5x - 2x = 12 + 3.", "Example: Solve 4x - 2 = 2x + 8", "2x = 10 ➔ x = 5"),
        QuestionItem("L2_H2", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 3(x + 4) = 27", 5.0, "Expand: 3x + 12 = 27. 3x = 15. x = 5", "Distribute 3 or divide both sides by 3.", "Example: Solve 2(x + 5) = 20", "x + 5 = 10 ➔ x = 5"),
        QuestionItem("L2_H3", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 4(2x - 1) = 36", 5.0, "Divide by 4: 2x - 1 = 9. 2x = 10. x = 5", "Divide by 4 first: 2x - 1 = 9.", "Example: Solve 3(3x - 2) = 21", "3x - 2 = 7 ➔ x = 3"),
        QuestionItem("L2_H4", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 7x + 3 = 3x + 35", 8.0, "Subtract 3x: 4x + 3 = 35. 4x = 32. x = 8", "Subtract 3x from 7x.", "Example: Solve 6x + 4 = 2x + 20", "4x = 16 ➔ x = 4"),
        QuestionItem("L2_H5", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: (3x - 5) / 4 = 7", 11.0, "Multiply by 4: 3x - 5 = 28. 3x = 33. x = 11", "Multiply by 4, then add 5.", "Example: Solve (2x - 4) / 3 = 6", "2x - 4 = 18 ➔ x = 11"),
        QuestionItem("L2_H6", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 5(x - 2) = 3x + 6", 8.0, "Expand: 5x - 10 = 3x + 6. 2x = 16. x = 8", "Distribute 5: 5x - 10 = 3x + 6.", "Example: Solve 4(x - 1) = 2x + 10", "4x - 4 = 2x + 10 ➔ 2x = 14 ➔ x = 7"),
        QuestionItem("L2_H7", 2, "Basic Algebra", "Hard", 0.8, "Solve for x: 9x - 14 = 4x + 21", 7.0, "Subtract 4x: 5x - 14 = 21. 5x = 35. x = 7", "Subtract 4x, add 14.", "Example: Solve 7x - 10 = 3x + 14", "4x = 24 ➔ x = 6"),
    ],

    # =========================================================================
    # KC 3: ADVANCED ALGEBRA (Quadratics, Exponent Rules, Systems)
    # =========================================================================
    3: [
        # Easy (0.2)
        QuestionItem("Q3_E1", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: x^2 - 4 = 0", 2.0, "x^2 = 4 ➔ x = √4 = 2", "Add 4 to both sides and take square root.", "Example: Solve x^2 - 9 = 0", "x^2 = 9 ➔ x = 3"),
        QuestionItem("Q3_E2", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: x^2 = 25", 5.0, "x = √25 = 5", "Take positive square root of 25.", "Example: Solve x^2 = 16", "x = √16 = 4"),
        QuestionItem("Q3_E3", 3, "Advanced Algebra", "Easy", 0.2, "Simplify exponent: x^3 * x^4 = x^?", 7.0, "Product rule: 3 + 4 = 7", "Add the exponents: 3 + 4.", "Example: x^2 * x^5", "2 + 5 = 7"),
        QuestionItem("Q3_E4", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: x^2 - 36 = 0", 6.0, "x^2 = 36 ➔ x = 6", "Add 36 to both sides.", "Example: Solve x^2 - 49 = 0", "x = √49 = 7"),
        QuestionItem("Q3_E5", 3, "Advanced Algebra", "Easy", 0.2, "Simplify exponent: (x^3)^2 = x^?", 6.0, "Power rule: 3 * 2 = 6", "Multiply the exponents: 3 * 2.", "Example: (x^4)^3", "4 * 3 = 12"),
        QuestionItem("Q3_E6", 3, "Advanced Algebra", "Easy", 0.2, "Solve for positive x: 2x^2 = 18", 3.0, "x^2 = 9 ➔ x = 3", "Divide by 2 first: x^2 = 9.", "Example: Solve 3x^2 = 48", "x^2 = 16 ➔ x = 4"),
        QuestionItem("Q3_E7", 3, "Advanced Algebra", "Easy", 0.2, "Simplify exponent: x^8 ÷ x^3 = x^?", 5.0, "Quotient rule: 8 - 3 = 5", "Subtract exponents: 8 - 3.", "Example: x^9 ÷ x^4", "9 - 4 = 5"),
        # Medium (0.5)
        QuestionItem("Q3_M1", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive x: x^2 - 5x + 6 = 0", 3.0, "Factoring (x - 2)(x - 3) = 0 ➔ x = 2 or x = 3. Larger root = 3", "Factor into (x - 2)(x - 3) = 0.", "Example: x^2 - 7x + 12 = 0", "(x - 3)(x - 4) = 0 ➔ roots 3, 4"),
        QuestionItem("Q3_M2", 3, "Advanced Algebra", "Medium", 0.5, "Solve system for x: x + y = 10, x - y = 4", 7.0, "Add equations: 2x = 14 ➔ x = 7", "Add the two equations to eliminate y.", "Example: x + y = 8, x - y = 2", "2x = 10 ➔ x = 5"),
        QuestionItem("Q3_M3", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive x: x^2 - 9x + 20 = 0", 5.0, "Factoring (x - 4)(x - 5) = 0 ➔ roots 4, 5. Larger root = 5", "Find numbers that multiply to 20 and add to -9.", "Example: x^2 - 6x + 8 = 0", "(x - 2)(x - 4) = 0 ➔ roots 2, 4"),
        QuestionItem("Q3_M4", 3, "Advanced Algebra", "Medium", 0.5, "Solve system for y: 2x + y = 11, x = 3", 5.0, "Substitute x = 3: 2(3) + y = 11 ➔ 6 + y = 11 ➔ y = 5", "Substitute x = 3 into first equation.", "Example: 3x + y = 14, x = 4", "12 + y = 14 ➔ y = 2"),
        QuestionItem("Q3_M5", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive x: 3x^2 - 12 = 0", 2.0, "3x^2 = 12 ➔ x^2 = 4 ➔ x = 2", "Divide by 3: x^2 = 4.", "Example: Solve 5x^2 - 45 = 0", "x^2 = 9 ➔ x = 3"),
        QuestionItem("Q3_M6", 3, "Advanced Algebra", "Medium", 0.5, "Solve for positive x: x^2 - 8x + 15 = 0", 5.0, "(x - 3)(x - 5) = 0 ➔ roots 3, 5. Larger root = 5", "Factor into (x - 3)(x - 5).", "Example: x^2 - 5x + 4 = 0", "(x - 1)(x - 4) = 0 ➔ roots 1, 4"),
        QuestionItem("Q3_M7", 3, "Advanced Algebra", "Medium", 0.5, "Solve system for x: 2x + 3y = 13, y = 3", 2.0, "2x + 3(3) = 13 ➔ 2x + 9 = 13 ➔ 2x = 4 ➔ x = 2", "Substitute y = 3.", "Example: 3x + 2y = 16, y = 2", "3x + 4 = 16 ➔ x = 4"),
        # Hard (0.8)
        QuestionItem("Q3_H1", 3, "Advanced Algebra", "Hard", 0.8, "Solve quadratic for positive root x: x^2 - 2x - 15 = 0", 5.0, "(x - 5)(x + 3) = 0 ➔ roots 5, -3. Positive root = 5", "Factor into (x - 5)(x + 3) = 0.", "Example: x^2 - x - 12 = 0", "(x - 4)(x + 3) = 0 ➔ positive root 4"),
        QuestionItem("Q3_H2", 3, "Advanced Algebra", "Hard", 0.8, "Solve system for x: 3x + 2y = 16, 2x - 2y = 4", 4.0, "Add equations: 5x = 20 ➔ x = 4", "Add both equations to eliminate y.", "Example: 4x + 3y = 25, x - 3y = 0", "5x = 25 ➔ x = 5"),
        QuestionItem("Q3_H3", 3, "Advanced Algebra", "Hard", 0.8, "Solve quadratic for positive root x: 2x^2 - 7x + 3 = 0", 3.0, "(2x - 1)(x - 3) = 0 ➔ roots 0.5, 3. Larger root = 3", "Use quadratic formula or factor (2x - 1)(x - 3).", "Example: 2x^2 - 5x + 2 = 0", "(2x - 1)(x - 2) = 0 ➔ roots 0.5, 2"),
        QuestionItem("Q3_H4", 3, "Advanced Algebra", "Hard", 0.8, "Solve system for x: 5x + 4y = 22, 3x + 4y = 14", 4.0, "Subtract second from first: 2x = 8 ➔ x = 4", "Subtract second equation from first.", "Example: 4x + 2y = 18, 2x + 2y = 10", "2x = 8 ➔ x = 4"),
        QuestionItem("Q3_H5", 3, "Advanced Algebra", "Hard", 0.8, "Solve quadratic for positive root x: x^2 - 6x - 16 = 0", 8.0, "(x - 8)(x + 2) = 0 ➔ roots 8, -2. Positive root = 8", "Factor into (x - 8)(x + 2) = 0.", "Example: x^2 - 3x - 10 = 0", "(x - 5)(x + 2) = 0 ➔ positive root 5"),
        QuestionItem("Q3_H6", 3, "Advanced Algebra", "Hard", 0.8, "Solve system for y: 4x + y = 19, 2x + y = 11", 3.0, "Subtract: 2x = 8 ➔ x = 4. 2(4) + y = 11 ➔ y = 3", "Subtract equations to find x = 4, then solve for y.", "Example: 3x + y = 13, x + y = 7", "2x = 6 ➔ x = 3, y = 4"),
        QuestionItem("Q3_H7", 3, "Advanced Algebra", "Hard", 0.8, "Solve quadratic for positive root x: 3x^2 - 10x + 3 = 0", 3.0, "(3x - 1)(x - 3) = 0 ➔ roots 1/3, 3. Larger root = 3", "Factor into (3x - 1)(x - 3) = 0.", "Example: 3x^2 - 7x + 2 = 0", "(3x - 1)(x - 2) = 0 ➔ roots 1/3, 2"),
    ],
}


def get_curriculum_question(
    kc_idx: int, action_idx: int, seed: Optional[int] = None
) -> QuestionItem:
    """Retrieves a curriculum question item matching the specified KC and Action difficulty.

    Args:
        kc_idx: Target Knowledge Component index in {0, 1, 2, 3}.
        action_idx: Pedagogical Action index in {0: Easy, 1: Medium, 2: Hard, 3: Worked Example, 4: Hint}.
        seed: Optional random seed for reproducible item selection.

    Returns:
        QuestionItem: Selected question item with prompt, answer, explanation, and hints.

    Raises:
        ValueError: If kc_idx is not in {0, 1, 2, 3}.
    """
    if kc_idx not in (0, 1, 2, 3):
        raise ValueError(f"kc_idx must be in {{0, 1, 2, 3}}, got {kc_idx}")

    items = QUESTION_BANK[kc_idx]
    rng = random.Random(seed) if seed is not None else random

    # Filter items by action difficulty
    if action_idx == 0:  # Easy
        candidates = [item for item in items if item.difficulty == "Easy"]
    elif action_idx == 1:  # Medium
        candidates = [item for item in items if item.difficulty == "Medium"]
    elif action_idx in (2, 3, 4):  # Hard, Worked Example, or Hint
        candidates = [item for item in items if item.difficulty == "Hard"]
    else:
        candidates = items

    if not candidates:
        candidates = items

    return rng.choice(candidates)
