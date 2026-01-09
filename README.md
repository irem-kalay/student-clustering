# Student Clustering Project

This repository consists of the process of creating vectors from student transcript data (Excel files) to transform them into **Numeric Feature Vectors** suitable for machine learning algorithms, specifically for **Clustering** tasks.

## 🚀 Features

- **Data Cleaning:** Reads raw Excel transcript files and handles missing or corrupted data.
- **Normalization:** Unifies course codes to handle curriculum changes (e.g., treats `MAT103E` (English) and `MAT103` (Turkish) as the same course to ensure consistency).
- **Grade Conversion:** Converts letter grades (AA, BA, CC, etc.) into numeric values on a 4.0 scale.
- **Vectorization:** Generates a comprehensive feature vector for each student based on their academic performance.
- **Verification:** Includes a bi-directional verification script to cross-check the generated vectors against the original Excel files to ensure data integrity.

## 🛠️ Installation
Follow these steps to set up the project locally.

### Prerequisites
- Python 3.8+
- Git

### Step-by-Step Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/irem-kalay/student-clustering.git](https://github.com/irem-kalay/student-clustering.git)
   cd student-clustering

Create a Virtual Environment:

2. **Create Virtual Environment:**
    ```bash
    # For Mac/Linux:
    python3 -m venv venv
    source venv/bin/activate

    # For Windows:
    python -m venv venv
    venv\Scripts\activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt