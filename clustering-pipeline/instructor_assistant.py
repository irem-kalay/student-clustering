# -*- coding: utf-8 -*-
"""
INSTRUCTOR DASHBOARD: Data-Driven Course Design Tool

This module helps instructors:
1. Select a course from the curriculum catalog
2. Analyze their current class composition (cluster distribution)
3. Get AI-powered recommendations for course delivery, assignments, and strategies
   tailored to their specific student mix
"""

import google.generativeai as genai
import pandas as pd
import json
import os

# --- 1. AYARLAR ---
API_KEY = "AIzaSyBDz-EAQmM8JRsFTntoSRIbaLX-sgZGCzM"

genai.configure(api_key=API_KEY)

# Dosya Yolları
CLUSTERS_CSV = "clustering-pipeline/results/final_student_clusters_no_naming.csv"
PERSONAS_JSON = "clustering-pipeline/results/AgenticAI/cluster_personas.json"
COURSE_CATALOG = "DersDenklikleri/dersdenklikleri.csv"

# --- 2. COURSE METADATA DEFINITIONS ---
COURSE_PROFILES = {
    'FIZ 101': {
        'name': 'Physics I - Mechanics',
        'category': 'Basic Sciences',
        'level': 'Introductory',
        'prerequisites': [],
        'math_intensity': 'High',
        'practical_component': 'Medium (Lab)',
        'best_for_clusters': ['Excellent performers', 'Good technical foundation'],
        'challenge': 'High mathematics content, Abstract concepts'
    },
    'BLG 101': {
        'name': 'Introduction to Programming',
        'category': 'Software Practice',
        'level': 'Introductory',
        'prerequisites': [],
        'math_intensity': 'Low-Medium',
        'practical_component': 'High (Coding)',
        'best_for_clusters': ['Software developers', 'Algorithm enthusiasts'],
        'challenge': 'Syntax learning, Problem decomposition'
    },
    'MAT 103': {
        'name': 'Calculus I',
        'category': 'Mathematics',
        'level': 'Introductory',
        'prerequisites': [],
        'math_intensity': 'Very High',
        'practical_component': 'Low',
        'best_for_clusters': ['Mathematical & Computational Focus'],
        'challenge': 'Abstract thinking, Proofs'
    },
    'BLG 210': {
        'name': 'Discrete Mathematics',
        'category': 'Algorithm Theory',
        'level': 'Intermediate',
        'prerequisites': ['MAT 103'],
        'math_intensity': 'Very High',
        'practical_component': 'Low',
        'best_for_clusters': ['Algorithmic Problem Solvers'],
        'challenge': 'Formal proofs, Set theory'
    },
    'BLG 231': {
        'name': 'Digital Design',
        'category': 'Hardware',
        'level': 'Intermediate',
        'prerequisites': ['BLG 101'],
        'math_intensity': 'Medium',
        'practical_component': 'High (Circuit design, Simulation)',
        'best_for_clusters': ['Hardware & Design Specialists', 'Systems-focused'],
        'challenge': 'Logic design, Circuit complexity'
    },
    'BLG 223': {
        'name': 'Data Structures',
        'category': 'Software Practice & Algorithm Theory',
        'level': 'Intermediate',
        'prerequisites': ['BLG 101', 'BLG 210'],
        'math_intensity': 'Medium-High',
        'practical_component': 'High (Implementation)',
        'best_for_clusters': ['Software developers', 'Algorithm enthusiasts'],
        'challenge': 'Complexity analysis, Abstract data types'
    },
    'BLG 252': {
        'name': 'Systems Programming',
        'category': 'Systems',
        'level': 'Intermediate',
        'prerequisites': ['BLG 223'],
        'math_intensity': 'Medium',
        'practical_component': 'Very High (Low-level coding)',
        'best_for_clusters': ['Systems & Infrastructure Focused'],
        'challenge': 'Memory management, Low-level abstractions'
    },
    'BLG 335': {
        'name': 'Algorithm Analysis',
        'category': 'Algorithm Theory',
        'level': 'Advanced',
        'prerequisites': ['BLG 210', 'BLG 223'],
        'math_intensity': 'Very High',
        'practical_component': 'Medium',
        'best_for_clusters': ['Algorithmic Problem Solvers'],
        'challenge': 'Proof techniques, NP-completeness'
    },
    'ING 100': {
        'name': 'English I',
        'category': 'Language & Communication',
        'level': 'Introductory',
        'prerequisites': [],
        'math_intensity': 'None',
        'practical_component': 'High (Speaking, Writing)',
        'best_for_clusters': ['Communication & Language Strong'],
        'challenge': 'Language proficiency, Cross-cultural communication'
    },
}

# --- 3. VERİ YÜKLEME ---
def load_data():
    """Load clustering data and personas"""
    try:
        df_all_students = pd.read_csv(CLUSTERS_CSV)
        df_all_students['Student_ID'] = df_all_students['Student_ID'].astype(str)
        
        with open(PERSONAS_JSON, 'r', encoding='utf-8') as f:
            personas_data = json.load(f)
        
        print("✅ Student clustering data loaded.")
        return df_all_students, personas_data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit()

def load_course_catalog():
    """Load course catalog from CSV"""
    try:
        df_courses = pd.read_csv(COURSE_CATALOG)
        df_courses['Ders Kodu'] = df_courses['Ders Kodu'].str.strip()
        courses = df_courses['Ders Kodu'].unique().tolist()  # Convert to list
        print(f"✅ Course catalog loaded: {len(courses)} courses available.")
        return courses
    except Exception as e:
        print(f"❌ Error loading course catalog: {e}")
        return []

# --- 4. SINIF ANALİZ FONKSİYONLARI ---

def analyze_class_composition(student_id_list, df_all_students, personas_data):
    """Analyze class composition in terms of cluster distribution"""
    class_df = df_all_students[df_all_students['Student_ID'].isin(student_id_list)]
    
    if class_df.empty:
        return None, None
    
    total_students = len(class_df)
    cluster_counts = class_df['Cluster'].value_counts().sort_values(ascending=False)
    
    summary_text = f"--- CLASS PROFILE ANALYSIS (Total: {total_students} Students) ---\n\n"
    summary_text += "CLUSTER COMPOSITION:\n"
    
    cluster_info = {}
    for cid in cluster_counts.index:
        count = cluster_counts[cid]
        percentage = (count / total_students) * 100
        
        p_info = personas_data["clusters"].get(f"cluster_{cid}")
        
        if p_info:
            first_line = p_info['description'].split('\n')[0]
            
            cluster_info[int(cid)] = {
                'name': first_line,
                'size': int(count),
                'percentage': round(percentage, 1),
                'strengths': p_info['strengths'][:3],
                'weaknesses': p_info['weaknesses'][:2],
                'performance': p_info['performance_level'],
                'gender': {
                    'male': p_info['gender_distribution']['male_percentage'],
                    'female': p_info['gender_distribution']['female_percentage']
                }
            }
            
            summary_text += f"\n👥 Cluster {cid} ({percentage:.1f}%, n={count})\n"
            summary_text += f"   {first_line}\n"
            summary_text += f"   Performance: {p_info['performance_level']}\n"
            summary_text += f"   Strengths: {', '.join(p_info['strengths'][:2])}\n"
            summary_text += f"   Weaknesses: {', '.join(p_info['weaknesses'][:2])}\n"
    
    return summary_text, cluster_info

# --- 5. COURSE-SPECIFIC RECOMMENDATIONS ---

def get_course_recommendations(course_code, class_analysis, cluster_info, personas_data):
    """Get AI-powered recommendations specific to the selected course and class composition"""
    
    course_info = COURSE_PROFILES.get(course_code, {
        'name': course_code,
        'category': 'Elective',
        'level': 'Unknown',
        'math_intensity': 'Unknown',
        'practical_component': 'Unknown'
    })
    
    system_prompt = f"""
    You are an experienced University Professor and Instructional Designer specializing in Computer Science and Engineering Education.
    
    COURSE DETAILS:
    - Course Code: {course_code}
    - Course Name: {course_info.get('name', 'N/A')}
    - Category: {course_info.get('category', 'N/A')}
    - Level: {course_info.get('level', 'N/A')}
    - Math Intensity: {course_info.get('math_intensity', 'N/A')}
    - Practical Component: {course_info.get('practical_component', 'N/A')}
    - Main Challenge: {course_info.get('challenge', 'N/A')}
    
    CLASS COMPOSITION:
    {class_analysis}
    
    Your mission is to provide detailed, actionable recommendations for teaching this specific course to this specific class mix.
    Consider the strengths and weaknesses of each dominant cluster.
    Provide concrete strategies, not generic advice.
    """
    
    return system_prompt, course_info

def ask_instructor_agent(class_analysis_text, instructor_prompt):
    """
    Sınıf analizini LLM'e verip hocanın isteğini yerine getirir.
    """
    
    system_prompt = f"""
    You are an expert Instructional Designer and Teaching Assistant AI.
    You are assisting a University Professor who wants to adapt their course content tailored to the specific profile of their current class.
    
    HERE IS THE ANALYSIS OF THE CURRENT CLASS:
    {class_analysis_text}
    
    YOUR MISSION:
    1. Analyze the 'Dominant Student Groups'. Look at their Strengths and Weaknesses.
    2. Respond to the Professor's request by adapting the content to fit this mix of students.
    3. If the class is mostly "Hardware Specialists", suggest practical, low-level implementation projects.
    4. If the class is "Social Leaders" but weak in Math, suggest group projects with less focus on heavy derivation and more on application/presentation.
    5. Be specific, academic, and creative.
    """
    
    # --- DEĞİŞİKLİK BURADA ---
    # Model ismini tırnak içinde ve tam olarak bu şekilde yazmalısın:
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    full_prompt = f"{system_prompt}\n\nPROFESSOR'S REQUEST: {instructor_prompt}\nAI SUGGESTION:"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        # Hata mesajını daha net görelim
        return f"API Hatası (Beklenmedik): {e}"

# --- 6. MENU INTERFACE ---

def select_course(courses):
    """Interactive course selection"""
    print("\n" + "="*70)
    print("📚 SELECT COURSE FROM CATALOG")
    print("="*70)
    
    courses_list = sorted(list(set(courses)))
    popular_courses = [c for c in courses_list if c in COURSE_PROFILES.keys()]
    other_courses = [c for c in courses_list if c not in COURSE_PROFILES.keys()]
    
    print("\n🌟 POPULAR COURSES (With detailed profiles):")
    for i, course in enumerate(popular_courses, 1):
        prof = COURSE_PROFILES[course]
        print(f"   {i:2d}. {course:12s} - {prof['name']}")
    
    print(f"\n📖 OTHER COURSES ({len(other_courses)} available):")
    for i, course in enumerate(other_courses, 1):
        if i <= 10:
            print(f"   {i+len(popular_courses):2d}. {course}")
    
    if len(other_courses) > 10:
        print(f"   ... and {len(other_courses) - 10} more courses")
    
    print("\n   Type 'list' to see all courses")
    print("   Type 'search <code>' to search for a course")
    
    while True:
        choice = input("\nEnter course code or command: ").strip().upper()
        
        if choice == 'LIST':
            for i, course in enumerate(courses_list, 1):
                print(f"{i:3d}. {course}")
            continue
        
        if choice.startswith('SEARCH '):
            search_term = choice.replace('SEARCH ', '').strip()
            matching = [c for c in courses_list if search_term in c]
            if matching:
                print(f"\nMatching courses: {', '.join(matching)}")
            else:
                print("No courses found matching that search term.")
            continue
        
        if choice in courses_list:
            return choice
        else:
            print(f"❌ Course '{choice}' not found. Please try again.")

def select_class_students(df_all_students):
    """Get class roster from user"""
    print("\n" + "="*70)
    print("📝 ENTER CLASS ROSTER")
    print("="*70)
    print("\nOptions:")
    print("1. Upload student IDs (paste comma-separated)")
    print("2. Use sample class (random 40 students)")
    print("3. Use specific cluster (e.g., 'cluster 3')")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        ids_input = input("Enter student IDs (comma-separated): ").strip()
        student_ids = [s.strip() for s in ids_input.split(',')]
        return student_ids
    
    elif choice == '2':
        sample_ids = df_all_students.sample(n=min(40, len(df_all_students)), random_state=42)['Student_ID'].tolist()
        print(f"✅ Selected {len(sample_ids)} random students")
        return sample_ids
    
    elif choice == '3':
        cluster_num = input("Enter cluster number (0-19): ").strip()
        try:
            cluster_num = int(cluster_num)
            cluster_students = df_all_students[df_all_students['Cluster'] == cluster_num]['Student_ID'].tolist()
            print(f"✅ Selected {len(cluster_students)} students from Cluster {cluster_num}")
            return cluster_students
        except:
            print("❌ Invalid cluster number")
            return None
    
    else:
        print("❌ Invalid choice")
        return None

# --- 7. MAIN INTERFACE ---

def main():
    df_all_students, personas_data = load_data()
    courses = load_course_catalog()
    
    if not courses:
        print("❌ Could not load course catalog")
        return
    
    print("\n" + "="*70)
    print("👨‍🏫 INSTRUCTOR DASHBOARD: DATA-DRIVEN COURSE DESIGN")
    print("="*70)
    print("\nThis tool helps you design courses tailored to your class composition.")
    print("Using AI + student cluster analysis for targeted recommendations.\n")
    
    # Step 1: Course selection
    selected_course = select_course(courses)
    print(f"\n✅ Selected Course: {selected_course}")
    
    # Step 2: Class roster
    student_ids = select_class_students(df_all_students)
    if not student_ids:
        print("❌ No students selected")
        return
    
    # Step 3: Analyze class
    class_analysis, cluster_info = analyze_class_composition(student_ids, df_all_students, personas_data)
    
    if not class_analysis:
        print("❌ Could not analyze class")
        return
    
    print("\n" + class_analysis)
    
    # Step 4: Get recommendations
    system_prompt, course_info = get_course_recommendations(
        selected_course, class_analysis, cluster_info, personas_data
    )
    
    print("\n" + "="*70)
    print("🤖 COURSE PLANNING OPTIONS")
    print("="*70)
    print("\n1. 📊 Course Structure & Schedule Recommendation")
    print("2. 📝 Assignment & Project Design")
    print("3. 🎯 Teaching Strategy & Pedagogy")
    print("4. ❓ Exam Question Design")
    print("5. 💡 Address Specific Weakness")
    print("6. 🎓 Learning Outcomes Adaptation")
    print("7. 💬 Custom Question")
    print("q. Quit\n")
    
    while True:
        choice = input("Select option (1-7, q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Thank you for using Instructor Dashboard!")
            break
        
        # Prepare data for prompts
        # Find most common weakness across all clusters
        all_weaknesses = []
        for info in cluster_info.values():
            all_weaknesses.extend(info['weaknesses'])
        
        most_common_weakness = max(set(all_weaknesses), key=all_weaknesses.count) if all_weaknesses else 'general proficiency gaps'
        
        prompts = {
            '1': f"""Design a detailed 14-week course structure and weekly schedule for {selected_course} ({course_info.get('name', 'course')}).
                     Consider the class composition: which weeks should focus on fundamentals vs advanced topics?
                     When should you introduce practical projects? When to assess understanding?""",
            
            '2': f"""Design 3-4 major assignments/projects for {selected_course} that:
                     - Engage all cluster types in this class
                     - Build in complexity progressively
                     - Address the weaknesses of the majority groups
                     - Can be completed by both strong and struggling students
                     For each assignment, explain its pedagogical purpose and how it fits the class profile.""",
            
            '3': f"""What specific teaching strategies should I use for {selected_course} given this class composition?
                     - Which students need extra scaffolding and why?
                     - What teaching methods work best for this mix?
                     - How should I structure lectures vs labs vs projects?
                     - How to keep advanced students engaged while supporting others?""",
            
            '4': f"""Design 2 exam questions for {selected_course}:
                     Question 1: A challenging question that plays to the strengths of the dominant group
                     Question 2: A fair question that tests both strong and struggling students appropriately
                     Include rubrics and explain how each question assesses the learning outcomes.""",
            
            '5': f"""The biggest weakness in this class is {most_common_weakness.replace('_', ' ')}.
                     Design a specific intervention, tutoring program, or workshop to address this gap in {selected_course}.
                     How can I make this support effective and accessible?""",
            
            '6': f"""Adapt the learning outcomes for {selected_course} based on this class composition.
                     Which outcomes might need to be adjusted (easier or harder)?
                     What are realistic expectations for each cluster?
                     How do I ensure learning objectives remain meaningful?""",
            
            '7': input("\nEnter your custom question for this course: ").strip()
        }
        
        if choice in prompts and prompts[choice]:
            print("\n⏳ Generating recommendations... (This may take a moment)\n")
            print("="*70)
            response = ask_instructor_agent(system_prompt, prompts[choice])
            print(response)
            print("="*70)
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()