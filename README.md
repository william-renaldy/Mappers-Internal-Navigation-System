# Mappers-Internal-Navigation-System

This project implements an indoor navigation system for users to find their way around large indoor spaces like campuses, malls, and hospitals. It utilizes Google Maps Services, GPS, and AI-based pathfinding to provide an efficient and user-friendly navigation experience.

## Features
* **Shortest Path Navigation:** Employs an AI algorithm to calculate the quickest route between origin and destination.
* **User-friendly Interface:** Allows users to enter their desired destination and view the path on their smartphone or tablet.

## Technologies

* Django (Web framework)
* Artificial Intelligence (Pathfinding algorithm)

## Setup Instructions

1. **Prerequisites:**
    * Python (3.x)
    * Django
  
## Running the Application

1. Migrate Django models:
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```
2. Run the development server:
    ```bash
    python manage.py runserver
    ```
3. Access the application in your web browser at http://127.0.0.1:8000/

## Usage

1. Open the application in your web browser.
2. Enter your desired destination within the building.
3. The system will calculate the shortest path and display it on the map.
