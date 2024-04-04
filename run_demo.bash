#!/bin/bash

# add current pwd to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Open a new terminal window and run the server
gnome-terminal -- bash -c "python3 demo/demo_server.py; exec bash"

# Wait for a bit to make sure the server has time to start up
sleep 1

# Run the client in this terminal window
python3 demo/demo_client.py