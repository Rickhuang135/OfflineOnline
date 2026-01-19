class Words:
    SAVEGAME = "save" # Start saving every frame of game
    CLOSEDISPLAYS = "end" # Gracefully shutdown each display
    id = "id" # Display identity for a child vgui process
    time_stamp = "time" # Time screenshot was taken (since start)
    shm_name = "shm_name" # Name of shared memory buffer
    shape = "shape" # Shape of shared memory buffer
    dtype = "dtype" # Datatype of shared memory buffer
    

class Actions:
    Jump = "0"
    Duck = "1"
    Nothing = "2"