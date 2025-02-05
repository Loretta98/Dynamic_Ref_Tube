/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_reactor_model_dynamic.h"

#define NX     REACTOR_MODEL_DYNAMIC_NX
#define NZ     REACTOR_MODEL_DYNAMIC_NZ
#define NU     REACTOR_MODEL_DYNAMIC_NU
#define NP     REACTOR_MODEL_DYNAMIC_NP


int main()
{
    int status = 0;
    reactor_model_dynamic_sim_solver_capsule *capsule = reactor_model_dynamic_acados_sim_solver_create_capsule();
    status = reactor_model_dynamic_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = reactor_model_dynamic_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = reactor_model_dynamic_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = reactor_model_dynamic_acados_get_sim_out(capsule);
    void *acados_sim_dims = reactor_model_dynamic_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;
    x_current[6] = 0.0;
    x_current[7] = 0.0;
    x_current[8] = 0.0;
    x_current[9] = 0.0;
    x_current[10] = 0.0;
    x_current[11] = 0.0;
    x_current[12] = 0.0;
    x_current[13] = 0.0;
    x_current[14] = 0.0;
    x_current[15] = 0.0;
    x_current[16] = 0.0;
    x_current[17] = 0.0;
    x_current[18] = 0.0;
    x_current[19] = 0.0;
    x_current[20] = 0.0;
    x_current[21] = 0.0;
    x_current[22] = 0.0;
    x_current[23] = 0.0;
    x_current[24] = 0.0;
    x_current[25] = 0.0;
    x_current[26] = 0.0;
    x_current[27] = 0.0;
    x_current[28] = 0.0;
    x_current[29] = 0.0;
    x_current[30] = 0.0;
    x_current[31] = 0.0;
    x_current[32] = 0.0;
    x_current[33] = 0.0;
    x_current[34] = 0.0;
    x_current[35] = 0.0;
    x_current[36] = 0.0;
    x_current[37] = 0.0;
    x_current[38] = 0.0;
    x_current[39] = 0.0;
    x_current[40] = 0.0;
    x_current[41] = 0.0;
    x_current[42] = 0.0;
    x_current[43] = 0.0;
    x_current[44] = 0.0;
    x_current[45] = 0.0;
    x_current[46] = 0.0;
    x_current[47] = 0.0;
    x_current[48] = 0.0;
    x_current[49] = 0.0;
    x_current[50] = 0.0;
    x_current[51] = 0.0;
    x_current[52] = 0.0;
    x_current[53] = 0.0;
    x_current[54] = 0.0;
    x_current[55] = 0.0;
    x_current[56] = 0.0;
    x_current[57] = 0.0;
    x_current[58] = 0.0;
    x_current[59] = 0.0;
    x_current[60] = 0.0;
    x_current[61] = 0.0;
    x_current[62] = 0.0;
    x_current[63] = 0.0;
    x_current[64] = 0.0;
    x_current[65] = 0.0;
    x_current[66] = 0.0;
    x_current[67] = 0.0;
    x_current[68] = 0.0;
    x_current[69] = 0.0;
    x_current[70] = 0.0;
    x_current[71] = 0.0;
    x_current[72] = 0.0;
    x_current[73] = 0.0;
    x_current[74] = 0.0;
    x_current[75] = 0.0;
    x_current[76] = 0.0;
    x_current[77] = 0.0;
    x_current[78] = 0.0;
    x_current[79] = 0.0;
    x_current[80] = 0.0;
    x_current[81] = 0.0;
    x_current[82] = 0.0;
    x_current[83] = 0.0;
    x_current[84] = 0.0;
    x_current[85] = 0.0;
    x_current[86] = 0.0;
    x_current[87] = 0.0;
    x_current[88] = 0.0;
    x_current[89] = 0.0;
    x_current[90] = 0.0;
    x_current[91] = 0.0;
    x_current[92] = 0.0;
    x_current[93] = 0.0;
    x_current[94] = 0.0;
    x_current[95] = 0.0;
    x_current[96] = 0.0;
    x_current[97] = 0.0;
    x_current[98] = 0.0;
    x_current[99] = 0.0;
    x_current[100] = 0.0;
    x_current[101] = 0.0;
    x_current[102] = 0.0;
    x_current[103] = 0.0;
    x_current[104] = 0.0;
    x_current[105] = 0.0;
    x_current[106] = 0.0;
    x_current[107] = 0.0;
    x_current[108] = 0.0;
    x_current[109] = 0.0;
    x_current[110] = 0.0;
    x_current[111] = 0.0;
    x_current[112] = 0.0;
    x_current[113] = 0.0;
    x_current[114] = 0.0;
    x_current[115] = 0.0;
    x_current[116] = 0.0;
    x_current[117] = 0.0;
    x_current[118] = 0.0;
    x_current[119] = 0.0;
    x_current[120] = 0.0;
    x_current[121] = 0.0;
    x_current[122] = 0.0;
    x_current[123] = 0.0;
    x_current[124] = 0.0;
    x_current[125] = 0.0;
    x_current[126] = 0.0;
    x_current[127] = 0.0;
    x_current[128] = 0.0;
    x_current[129] = 0.0;
    x_current[130] = 0.0;
    x_current[131] = 0.0;
    x_current[132] = 0.0;
    x_current[133] = 0.0;
    x_current[134] = 0.0;
    x_current[135] = 0.0;
    x_current[136] = 0.0;
    x_current[137] = 0.0;
    x_current[138] = 0.0;
    x_current[139] = 0.0;
    x_current[140] = 0.0;
    x_current[141] = 0.0;
    x_current[142] = 0.0;
    x_current[143] = 0.0;
    x_current[144] = 0.0;
    x_current[145] = 0.0;
    x_current[146] = 0.0;
    x_current[147] = 0.0;
    x_current[148] = 0.0;
    x_current[149] = 0.0;
    x_current[150] = 0.0;
    x_current[151] = 0.0;
    x_current[152] = 0.0;
    x_current[153] = 0.0;
    x_current[154] = 0.0;
    x_current[155] = 0.0;
    x_current[156] = 0.0;
    x_current[157] = 0.0;
    x_current[158] = 0.0;
    x_current[159] = 0.0;
    x_current[160] = 0.0;
    x_current[161] = 0.0;
    x_current[162] = 0.0;
    x_current[163] = 0.0;
    x_current[164] = 0.0;
    x_current[165] = 0.0;
    x_current[166] = 0.0;
    x_current[167] = 0.0;
    x_current[168] = 0.0;
    x_current[169] = 0.0;
    x_current[170] = 0.0;
    x_current[171] = 0.0;
    x_current[172] = 0.0;
    x_current[173] = 0.0;
    x_current[174] = 0.0;
    x_current[175] = 0.0;
    x_current[176] = 0.0;
    x_current[177] = 0.0;
    x_current[178] = 0.0;
    x_current[179] = 0.0;
    x_current[180] = 0.0;
    x_current[181] = 0.0;
    x_current[182] = 0.0;
    x_current[183] = 0.0;
    x_current[184] = 0.0;
    x_current[185] = 0.0;
    x_current[186] = 0.0;
    x_current[187] = 0.0;
    x_current[188] = 0.0;
    x_current[189] = 0.0;
    x_current[190] = 0.0;
    x_current[191] = 0.0;
    x_current[192] = 0.0;
    x_current[193] = 0.0;
    x_current[194] = 0.0;
    x_current[195] = 0.0;
    x_current[196] = 0.0;
    x_current[197] = 0.0;
    x_current[198] = 0.0;
    x_current[199] = 0.0;
    x_current[200] = 0.0;
    x_current[201] = 0.0;
    x_current[202] = 0.0;
    x_current[203] = 0.0;
    x_current[204] = 0.0;
    x_current[205] = 0.0;
    x_current[206] = 0.0;
    x_current[207] = 0.0;
    x_current[208] = 0.0;
    x_current[209] = 0.0;
    x_current[210] = 0.0;
    x_current[211] = 0.0;
    x_current[212] = 0.0;
    x_current[213] = 0.0;
    x_current[214] = 0.0;
    x_current[215] = 0.0;
    x_current[216] = 0.0;
    x_current[217] = 0.0;
    x_current[218] = 0.0;
    x_current[219] = 0.0;
    x_current[220] = 0.0;
    x_current[221] = 0.0;
    x_current[222] = 0.0;
    x_current[223] = 0.0;
    x_current[224] = 0.0;
    x_current[225] = 0.0;
    x_current[226] = 0.0;
    x_current[227] = 0.0;
    x_current[228] = 0.0;
    x_current[229] = 0.0;
    x_current[230] = 0.0;
    x_current[231] = 0.0;
    x_current[232] = 0.0;
    x_current[233] = 0.0;
    x_current[234] = 0.0;
    x_current[235] = 0.0;
    x_current[236] = 0.0;
    x_current[237] = 0.0;
    x_current[238] = 0.0;
    x_current[239] = 0.0;
    x_current[240] = 0.0;
    x_current[241] = 0.0;
    x_current[242] = 0.0;
    x_current[243] = 0.0;
    x_current[244] = 0.0;
    x_current[245] = 0.0;
    x_current[246] = 0.0;
    x_current[247] = 0.0;
    x_current[248] = 0.0;
    x_current[249] = 0.0;
    x_current[250] = 0.0;
    x_current[251] = 0.0;
    x_current[252] = 0.0;
    x_current[253] = 0.0;
    x_current[254] = 0.0;
    x_current[255] = 0.0;
    x_current[256] = 0.0;
    x_current[257] = 0.0;
    x_current[258] = 0.0;
    x_current[259] = 0.0;
    x_current[260] = 0.0;
    x_current[261] = 0.0;
    x_current[262] = 0.0;
    x_current[263] = 0.0;
    x_current[264] = 0.0;
    x_current[265] = 0.0;
    x_current[266] = 0.0;
    x_current[267] = 0.0;
    x_current[268] = 0.0;
    x_current[269] = 0.0;
    x_current[270] = 0.0;
    x_current[271] = 0.0;
    x_current[272] = 0.0;
    x_current[273] = 0.0;
    x_current[274] = 0.0;
    x_current[275] = 0.0;
    x_current[276] = 0.0;
    x_current[277] = 0.0;
    x_current[278] = 0.0;
    x_current[279] = 0.0;
    x_current[280] = 0.0;
    x_current[281] = 0.0;
    x_current[282] = 0.0;
    x_current[283] = 0.0;
    x_current[284] = 0.0;
    x_current[285] = 0.0;
    x_current[286] = 0.0;
    x_current[287] = 0.0;
    x_current[288] = 0.0;
    x_current[289] = 0.0;
    x_current[290] = 0.0;
    x_current[291] = 0.0;
    x_current[292] = 0.0;
    x_current[293] = 0.0;
    x_current[294] = 0.0;
    x_current[295] = 0.0;
    x_current[296] = 0.0;
    x_current[297] = 0.0;
    x_current[298] = 0.0;
    x_current[299] = 0.0;
    x_current[300] = 0.0;
    x_current[301] = 0.0;
    x_current[302] = 0.0;
    x_current[303] = 0.0;
    x_current[304] = 0.0;
    x_current[305] = 0.0;
    x_current[306] = 0.0;
    x_current[307] = 0.0;
    x_current[308] = 0.0;
    x_current[309] = 0.0;
    x_current[310] = 0.0;
    x_current[311] = 0.0;
    x_current[312] = 0.0;
    x_current[313] = 0.0;
    x_current[314] = 0.0;
    x_current[315] = 0.0;
    x_current[316] = 0.0;
    x_current[317] = 0.0;
    x_current[318] = 0.0;
    x_current[319] = 0.0;
    x_current[320] = 0.0;
    x_current[321] = 0.0;
    x_current[322] = 0.0;
    x_current[323] = 0.0;
    x_current[324] = 0.0;
    x_current[325] = 0.0;
    x_current[326] = 0.0;
    x_current[327] = 0.0;
    x_current[328] = 0.0;
    x_current[329] = 0.0;
    x_current[330] = 0.0;
    x_current[331] = 0.0;
    x_current[332] = 0.0;
    x_current[333] = 0.0;
    x_current[334] = 0.0;
    x_current[335] = 0.0;
    x_current[336] = 0.0;
    x_current[337] = 0.0;
    x_current[338] = 0.0;
    x_current[339] = 0.0;
    x_current[340] = 0.0;
    x_current[341] = 0.0;
    x_current[342] = 0.0;
    x_current[343] = 0.0;
    x_current[344] = 0.0;
    x_current[345] = 0.0;
    x_current[346] = 0.0;
    x_current[347] = 0.0;
    x_current[348] = 0.0;
    x_current[349] = 0.0;
    x_current[350] = 0.0;
    x_current[351] = 0.0;
    x_current[352] = 0.0;
    x_current[353] = 0.0;
    x_current[354] = 0.0;
    x_current[355] = 0.0;
    x_current[356] = 0.0;
    x_current[357] = 0.0;
    x_current[358] = 0.0;
    x_current[359] = 0.0;
    x_current[360] = 0.0;
    x_current[361] = 0.0;
    x_current[362] = 0.0;
    x_current[363] = 0.0;
    x_current[364] = 0.0;
    x_current[365] = 0.0;
    x_current[366] = 0.0;
    x_current[367] = 0.0;
    x_current[368] = 0.0;
    x_current[369] = 0.0;
    x_current[370] = 0.0;
    x_current[371] = 0.0;
    x_current[372] = 0.0;
    x_current[373] = 0.0;
    x_current[374] = 0.0;
    x_current[375] = 0.0;
    x_current[376] = 0.0;
    x_current[377] = 0.0;
    x_current[378] = 0.0;
    x_current[379] = 0.0;
    x_current[380] = 0.0;
    x_current[381] = 0.0;
    x_current[382] = 0.0;
    x_current[383] = 0.0;
    x_current[384] = 0.0;
    x_current[385] = 0.0;
    x_current[386] = 0.0;
    x_current[387] = 0.0;
    x_current[388] = 0.0;
    x_current[389] = 0.0;
    x_current[390] = 0.0;
    x_current[391] = 0.0;
    x_current[392] = 0.0;
    x_current[393] = 0.0;
    x_current[394] = 0.0;
    x_current[395] = 0.0;
    x_current[396] = 0.0;
    x_current[397] = 0.0;
    x_current[398] = 0.0;
    x_current[399] = 0.0;
    x_current[400] = 0.0;
    x_current[401] = 0.0;
    x_current[402] = 0.0;
    x_current[403] = 0.0;
    x_current[404] = 0.0;
    x_current[405] = 0.0;
    x_current[406] = 0.0;
    x_current[407] = 0.0;
    x_current[408] = 0.0;
    x_current[409] = 0.0;
    x_current[410] = 0.0;
    x_current[411] = 0.0;
    x_current[412] = 0.0;
    x_current[413] = 0.0;
    x_current[414] = 0.0;
    x_current[415] = 0.0;
    x_current[416] = 0.0;
    x_current[417] = 0.0;
    x_current[418] = 0.0;
    x_current[419] = 0.0;
    x_current[420] = 0.0;
    x_current[421] = 0.0;
    x_current[422] = 0.0;
    x_current[423] = 0.0;
    x_current[424] = 0.0;
    x_current[425] = 0.0;
    x_current[426] = 0.0;
    x_current[427] = 0.0;
    x_current[428] = 0.0;
    x_current[429] = 0.0;
    x_current[430] = 0.0;
    x_current[431] = 0.0;
    x_current[432] = 0.0;
    x_current[433] = 0.0;
    x_current[434] = 0.0;
    x_current[435] = 0.0;
    x_current[436] = 0.0;
    x_current[437] = 0.0;
    x_current[438] = 0.0;
    x_current[439] = 0.0;
    x_current[440] = 0.0;
    x_current[441] = 0.0;
    x_current[442] = 0.0;
    x_current[443] = 0.0;
    x_current[444] = 0.0;
    x_current[445] = 0.0;
    x_current[446] = 0.0;
    x_current[447] = 0.0;
    x_current[448] = 0.0;
    x_current[449] = 0.0;
    x_current[450] = 0.0;
    x_current[451] = 0.0;
    x_current[452] = 0.0;
    x_current[453] = 0.0;
    x_current[454] = 0.0;
    x_current[455] = 0.0;
    x_current[456] = 0.0;
    x_current[457] = 0.0;
    x_current[458] = 0.0;
    x_current[459] = 0.0;
    x_current[460] = 0.0;
    x_current[461] = 0.0;
    x_current[462] = 0.0;
    x_current[463] = 0.0;
    x_current[464] = 0.0;
    x_current[465] = 0.0;
    x_current[466] = 0.0;
    x_current[467] = 0.0;
    x_current[468] = 0.0;
    x_current[469] = 0.0;
    x_current[470] = 0.0;
    x_current[471] = 0.0;
    x_current[472] = 0.0;
    x_current[473] = 0.0;
    x_current[474] = 0.0;
    x_current[475] = 0.0;
    x_current[476] = 0.0;
    x_current[477] = 0.0;
    x_current[478] = 0.0;
    x_current[479] = 0.0;

  
    printf("main_sim: initial state not defined, should be in lbx_0, using zero vector.");


    // initial value for control input
    double u0[NU];

  
    double S_forw[NX*(NX+NU)];
  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = reactor_model_dynamic_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "S_forw", S_forw);

        printf("\nS_forw, %d\n", ii);
        for (int i = 0; i < NX; i++)
        {
            for (int j = 0; j < NX+NU; j++)
            {
                printf("%+.3e ", S_forw[j * NX + i]);
            }
            printf("\n");
        }
    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = reactor_model_dynamic_acados_sim_free(capsule);
    if (status) {
        printf("reactor_model_dynamic_acados_sim_free() returned status %d. \n", status);
    }

    reactor_model_dynamic_acados_sim_solver_free_capsule(capsule);

    return status;
}
