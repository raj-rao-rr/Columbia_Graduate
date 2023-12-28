\ Model cell_tower
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 174 x - 184 y + 6129 Constant + [ 8 x ^2 + 8 y ^2 ] / 2 
Subject To
 c0: - 10 x - 90 y + [ x ^2 + y ^2 ] <= -450
 c1: - 24 x - 42 y + [ x ^2 + y ^2 ] <= 1015
 c2: - 36 x - 10 y + [ x ^2 + y ^2 ] <= 1251
 c3: - 104 x - 42 y + [ x ^2 + y ^2 ] <= -1545
Bounds
 Constant = 1
End
