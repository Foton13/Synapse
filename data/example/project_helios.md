# Project Helios

Project Helios is an internal data platform built for real-time analytics.
It was designed by the engineering team in Q1 2026.

## Tech Stack

The backend is built with **Python** and **FastAPI**, while the frontend
uses **React** with **TypeScript**. Data pipelines are orchestrated by
**Apache Airflow** and the warehouse runs on **ClickHouse**.

## Architecture

- The API Gateway connects to the Auth Service for token validation.
- The Ingestion Service pushes events into a Kafka topic.
- The Analytics Engine consumes Kafka events and writes aggregated
  metrics to ClickHouse.

## Team

- Alice — backend lead
- Bob — data engineer
- Carol — frontend developer
