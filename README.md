# SatNav: Satellite Imagery-Driven Mapping & Routing Intelligence

SatNav integrates satellite imagery analysis with Deep learning to enhance urban planning and transportation optimization. By identifying structures, terrain types, and optimizing transport routes, SatNav aims to revolutionize navigation solutions purely from satellite imagery data.

## Overview

SatNav utilizes advanced technologies to process satellite data, employing Deep learning models to extract meaningful insights for urban infrastructure planning. This project focuses on data ingestion, preprocessing, model training, deployment, route optimization, and visualization.

## Technology Stack

- **Data Handling:** Apache Kafka, AWS S3
- **Data Processing:** AWS Glue, AWS EMR (Apache Spark)
- **Machine Learning:** TensorFlow
- **Containerization:** Docker, Kubernetes, AWS EKS
- **Visualization:** Leaflet

## Key Features

- **Data Processing:** Utilize Apache Kafka for real-time data streaming, AWS Glue for ETL processes, and Apache Spark on AWS EMR for preprocessing satellite images.
  
- **Machine Learning Models:** Develop a U-Net based architecture using TensorFlow to classify urban, rural, forest, and water terrains from satellite images. Implement the A-Star algorithm for route optimization based on environmental characteristics.

- **Deployment:** Manage model deployment using Docker containers and Kubernetes on AWS EKS, ensuring scalability and reliability.

- **Visualization:** Create interactive map visualizations for urban planning using Leaflet, integrating structure, terrain data, and optimized transport routes.

## Project Workflow

Our approach to building SatNav involves several key phases, each critical to achieving our goal of leveraging satellite imagery for advanced mapping and routing intelligence.

1. **Data Collection**
   - Gather satellite imagery data from diverse sources including the EuroSAT dataset, Google Earth Engine, and Sentinel Hub. This ensures comprehensive coverage and quality data inputs for our analyses.

2. **Data Preprocessing**
   - Employ AWS Glue for tasks like resizing, normalization, and metadata creation.
   - Use Apache Spark on AWS EMR to segment large images into manageable parts, optimizing them for downstream operations.

3. **Model Training**
   - Develop sophisticated machine learning models, particularly a U-Net architecture using TensorFlow.
   - Train models on high-performance AWS instances to classify terrain types such as urban, rural, forest, and water, crucial for our route optimization algorithms.

4. **Model Deployment and Inference**
   - Containerize trained models using Docker for seamless deployment.
   - Orchestrate model deployment with Kubernetes on AWS EKS to enable scalable and reliable real-time inference, supporting continuous monitoring of new satellite data streams.

5. **Route Optimization**
   - Represent geographical regions as graphs to facilitate route optimization.
   - Utilize graph-based algorithms like A-Star to determine the most efficient transport routes considering terrain types and structural data.

6. **Visualization and Analysis**
   - Create interactive maps using Leaflet to visualize optimized transport routes, terrain analysis, and structural insights.
   - Enhance urban planning and decision-making processes through comprehensive visualization capabilities.
