$log = "D:\marineai\scratch\mirror.log"
robocopy D:\marineai\classification-experiments\shards     N:\marineai\classification-experiments\shards     /MIR /R:2 /W:5 /MT:16 /NP /LOG+:$log
robocopy D:\marineai\classification-experiments\embeddings N:\marineai\classification-experiments\embeddings /MIR /R:2 /W:5 /MT:16 /NP /LOG+:$log