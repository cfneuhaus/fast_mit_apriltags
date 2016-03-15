/** Tag family with 12 distinct codes.
    bits: 16,  minimum hamming: 6,  minimum complexity: 1

    Max bits corrected       False positive rate
            0                  0.01831055 %
            1                  0.31127930 %
            2                  2.50854492 %

    Generation time: 0.120000 s

    Hamming distance between pairs of codes (accounting for rotation):

       0  0
       1  0
       2  0
       3  0
       4  0
       5  0
       6  30
       7  31
       8  1
       9  4
      10  0
      11  0
      12  0
      13  0
      14  0
      15  0
      16  0
**/
#pragma once

namespace AprilTags {

const unsigned long long t16h6[] =
  { 0xdf22L, 0xeaacL, 0x0d4aL, 0x130fL, 0x18d4L, 0x91fdL, 0x7a42L, 0xe1d3L, 0xa77aL, 0xf52cL, 0xec1aL, 0x3460L };

static const TagCodes tagCodes16h6 = TagCodes(16, 6, t16h6, sizeof(t16h6)/sizeof(t16h6[0]));

}
