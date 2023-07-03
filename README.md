# Towards Language Model-based Identification of Business Clusters

Repository for the Data Science 2023 course project. 

One of the most important decisions every business has to decide on is a physical location. Identifying the right location is crucial for effective marketing strategies to reach the intended customer base. It can also reduce transportation costs, and locating in a vibrant regional business cluster can significantly positively affect business development.
However, strategically acquiring \& analyzing data on regional economies, the affiliated companies, and their target market segments is very complex and labor-intensive. 
	
We propose to circumvent these challenges by leveraging state-of-the-art language models to transform publicly available company websites into machine-readable data points and applying clustering to make location recommendations for novel companies.
	
A qualitative evaluation of our methods being applied to data from the German municipality Saarland demonstrates that our methods can accurately cluster related companies and categorize novel companies into existing clusters.

## Executing the Dashboard

To execute the Dashboard there are two possibilities:

- [Local Build](#local): Install the dependencies on your machine and execute the project server directly on your machine.
- [Docker Build](#docker): Build the docker image with the provided Docker image and execute the project in a container.

### Local

When building locally, it is recommended to set up a [virtual environment](https://docs.python.org/3/library/venv.html).

To do so, clone this repository and navigate to its folder. Then execute

```shell
python3 -m venv env && source env/bin/activate && which python3
```

to create and activate a new environment. This command should print the path to the newly created virtual environment. Install all the relevant dependencies by using:

```shell
python3 -m pip install -r requirements.txt
```

Now start the dashboard server by executing

```shell
python3 main.py
```

If the Dashboard page does not open automatically, it can be reached at [http://localhost:8501/](http://localhost:8501/).

Afterward, deactivate the python virtual environment by executing

```shell
deactivate
```

### Docker

For convenience, we also include a Dockerfile. It can be built using:

```shell
docker build . -t data_science
```

Afterwards, it can be executed using:

```shell
docker run -p 8501:8501 data_science
```

In this case `-p 8501:8501` specifies to forward the image port `8501` to your local port `8501`. The Dashboard is then reachable under [http://localhost:8501/](http://localhost:8501/).
