#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for instructor_assistant.py - simulates user interactions
"""

import sys
sys.path.insert(0, 'clustering-pipeline')

from instructor_assistant import (
    load_data, 
    load_course_catalog, 
    analyze_class_composition,
    get_course_recommendations
)

print("\n" + "="*70)
print("🧪 TESTING INSTRUCTOR ASSISTANT FUNCTIONALITY")
print("="*70)

# Test 1: Load data
print("\n📦 TEST 1: Loading data...")
try:
    df_all_students, personas_data = load_data()
    print(f"✅ Data loaded: {len(df_all_students)} students")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Load course catalog
print("\n📚 TEST 2: Loading course catalog...")
try:
    courses = load_course_catalog()
    assert isinstance(courses, list), "Courses should be a list"
    assert len(courses) > 0, "Should have courses"
    print(f"✅ Courses loaded: {len(courses)} courses")
    print(f"   Sample: {courses[:3]}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 3: Analyze class composition
print("\n🎓 TEST 3: Analyzing class composition...")
try:
    # Get 30 random students
    sample_ids = df_all_students.sample(n=30, random_state=42)['Student_ID'].tolist()
    class_analysis, cluster_info = analyze_class_composition(
        sample_ids, 
        df_all_students, 
        personas_data
    )
    assert class_analysis is not None, "Class analysis should not be None"
    assert cluster_info is not None, "Cluster info should not be None"
    assert len(cluster_info) > 0, "Should have cluster info"
    print(f"✅ Class analyzed: {len(sample_ids)} students")
    print(f"   Clusters found: {list(cluster_info.keys())}")
    print(f"\n{class_analysis}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 4: Get course recommendations
print("\n🤖 TEST 4: Generating course recommendations...")
try:
    course_code = 'BLG 223'
    system_prompt, course_info = get_course_recommendations(
        course_code,
        class_analysis,
        cluster_info,
        personas_data
    )
    assert system_prompt is not None, "System prompt should not be None"
    assert course_info is not None, "Course info should not be None"
    print(f"✅ Recommendations generated for {course_code}")
    print(f"   Course name: {course_info.get('name', 'N/A')}")
    print(f"   Level: {course_info.get('level', 'N/A')}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 5: Test with another course (one without detailed profile)
print("\n🤖 TEST 5: Testing with unlisted course...")
try:
    course_code = 'BLG 337'  # This one is not in COURSE_PROFILES
    system_prompt, course_info = get_course_recommendations(
        course_code,
        class_analysis,
        cluster_info,
        personas_data
    )
    assert system_prompt is not None, "System prompt should not be None"
    print(f"✅ Recommendations generated for {course_code} (default profile)")
    print(f"   Category: {course_info.get('category', 'N/A')}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run: python clustering-pipeline/instructor_assistant.py")
print("to start the interactive interface.\n")
