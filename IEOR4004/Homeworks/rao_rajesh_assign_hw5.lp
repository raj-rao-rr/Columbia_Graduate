\ Model network
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 x-1_2 ^2 + 8 x-1_3 ^2 + 12 x-2_3 ^2 + 24 x-2_4 ^2 + 6 x-3_4 ^2 ] / 2 
Subject To
 c1: x-1_2 + x-1_3 = 710
 c2: x-1_2 - x-2_3 - x-2_4 = 0
 c3: x-1_3 + x-2_3 - x-3_4 = 0
 c4: x-2_4 + x-3_4 = 710
Bounds
End
