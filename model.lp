\ Model Initializing Model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  ub + 0 lb
Subject To
 R0: z1[0] + z1[1] + z1[2] + z1[3] + z1[4] + z1[5] = 3
 R1: z2[0] + z2[1] + z2[2] + z2[3] + z2[4] = 3
Bounds
 ub <= 1
 lb <= 1
Binaries
 z1[0] z1[1] z1[2] z1[3] z1[4] z1[5] z2[0] z2[1] z2[2] z2[3] z2[4]
End
