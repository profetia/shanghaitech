# Horras: Haikou Online Ride-hailing and Records Analysis System

Horras is a system for analyzing the data of online ride-hailing services in Haikou, China. It is designed to be used in classroom settings, mainly for teaching data analysis and visualization. The backend of Horras can be seen in [Horras-Backend](https://github.com/yanglinshu/horras-backend).

![](/media/preview.png)

## Installation

Horras is a web application built with [Vue.js](https://vuejs.org/).

### Development

To install Horras in a development environment, you need to have [Node.js](https://nodejs.org/) installed. Then, clone this repository and run the following commands in the root directory of the project:

```bash
npm install
npm run dev
```

The application will be running at [http://localhost:3000](http://localhost:3000).

### Production

Horras is recommended to be deployed with [Docker](https://www.docker.com/). To build the Docker image, run the following command in the root directory of the project:

```bash
docker-compose up -d
```

The application will be running at [http://localhost:6699](http://localhost:6699).

## Documentation

Documentation for Horras can be found in the [docs](/docs) directory.

## Misc

Posters and slides for Horras can be found in the [media](/media) directory.

## Backend

### Dependencies
Horras Backend is built on top of the following dependencies:
- [Fastapi](https://fastapi.tiangolo.com/)
- [PostgreSQL](https://www.postgresql.org/)
- [Prisma](https://www.prisma.io/)
- [Docker](https://www.docker.com/)

### Installation
Horras Backend runs its database in a Docker container. To install the database, run the following command:
```bash
docker-compose up -d
```

To install the backend, run the following commands:
```bash
pip install -r requirements.txt

prisma db push

uvicorn horras_backend.main:app --host 0.0.0.0
```

For compatibility reasons, running the backend in a Docker container is not recommended.

### Usage
The backend is a Fastapi project. To access the API documentation, go to `http://localhost:8000/docs`.

## References

Horras preprocesses its data from the [SARROH](https://github.com/xsjk/ARTS1422-Project).

## License

This repository is licensed under the [MIT License](/LICENSE).
