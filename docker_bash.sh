# Insufficient number of arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./run_docker.sh [run|exec|build|stop|remove]"
    exit 1
fi

case $1 in
    run)
        # Run the docker container
        docker run -v ./:/src/ --rm  -d -it  --name mm-am-container mm-am
        ;;
    exec)
        # Execute the models inside the docker container
        docker exec -it mm-am-container bash      
        ;;
    build)
        # Build the docker
        docker build ./ -t mm-am
        ;;
    stop)
        # Stop the docker container
        docker stop mm-am-container
        ;;
    remove)
        # Remove the docker container
        docker stop mm-am-container &&
        docker remove mm-am-container
        ;;
    *)
        # Invalid argument
        echo "Invalid argument"
        ;;
esac