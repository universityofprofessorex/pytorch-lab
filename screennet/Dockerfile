FROM continuumio/anaconda3:2022.10 AS build

WORKDIR /code

# Python shall not write the byte code to *.pyc files; they cannot be cached between
# runs of the container anyway, hence we save the required time and resources for that
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering output that is written to STDOUT/STDERR; this allows to
# monitor the output in real time
ENV PYTHONUNBUFFERED 1

COPY . .

RUN conda env update

# Execution stage
FROM continuumio/anaconda3:2022.10 AS execute

# Set environment variables
# Set the Pynguin version
# ENV PYNGUIN_VERSION "0.27.0"
# Pynguin requires to set the variable to show it that the user is aware that running
# Pynguin executes third-party code, which could cause arbitrary harm to the system.
# By setting the variable, the user acknowledges that they are aware of this.  In the
# container this is not too much of an issue (if we forget about breaking out of the
# container and similar things, of course).
# ENV PYNGUIN_DANGER_AWARE ""
# Python shall not write the byte code to *.pyc files; they cannot be cached between
# runs of the container anyway, hence we save the required time and resources for that
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering output that is written to STDOUT/STDERR; this allows to
# monitor the output in real time
ENV PYTHONUNBUFFERED 1
# Disable random hash seeding for reproducible runs
# ENV PYTHONHASHSEED 0

WORKDIR /code

COPY --from=build /code .

# ENTRYPOINT ["/pynguin/pynguin-docker.sh"]
ENTRYPOINT ["bash"]
CMD []
