from scipy import signal
z = [190.502904680020,
190.203931774876,
189.999907050239,
190.045684807362,
190.170516335920,
190.342380443370,
190.763535808895,
190.622949295906,
190.499930287680,
189.455639726728,
188.245201468606,
187.864897522889,
188.913742622113,
189.520751034066,
190.311846446995,
191.177348143329,
191.974531765581,
192.280894176698,
192.286982386020,
192.409350745922,
192.279546405168,
192.678068503974,
193.475252126226,
193.642515220523,
193.411070316494,
193.029046800205,
192.726588279035,
192.894362597016,
193.133057582377,
193.242738299949,
193.088162847981,
193.207742715063,
192.427197099967,
192.348979876377,
192.175628572756,
192.930427104150,
193.514755774504,
193.747037226379,
193.797137147372,
193.761072640238,
193.530603708695,
193.627178510015,
194.083561834828,
194.162290282103,
194.008737277502,
194.022586791839,
194.464702328392,
194.710600920203,
195.023702188967,
195.011851094483,
195.311381698192,
195.308872054654,
195.072500813310,
194.730771018265,
194.456290375052,
194.666775108054,
194.764093507459,
195.045080633917,
195.752474787377,
196.080122693684,
196.304828740066,
195.971371473718,
195.758237672538,
195.576009666775,
195.526420969466,
195.577125063903,
195.250546079844,
195.184505274899,
194.894594971418,
194.531068457499,
194.812427383000,
195.020216572942,
195.337593530697,
195.286238787935,
194.632290746851,
193.641260398754,
193.121950085979,
193.073383836037,
193.097411349166,
192.876005019287,
192.830738485848,
193.072826137473,
193.295952037924,
193.245898591811,
192.979365153135,
193.270158479342,
193.284937491286,
193.046428405447,
192.984709764372,
192.616721661942,
192.018078728447,
191.128038295301,
190.530557233815,
190.679788074546,
190.708556025468,
190.670307198959,
190.384533159827,
190.435655528187,
190.269554305898,
190.256169540363,
190.672584468095,
190.245526792769,
190.804061904541,
190.939722080216,
191.062694613561,
190.635636938235,
190.524050750569,
190.833155179625,
190.907654412790,
191.044337035832,
191.002230794256,
191.231816703072,
190.727982525445,
190.396895477994,
190.013617139936,
189.805316726309,
190.273969419529,
190.627317934656,
190.223869498536,
189.618255332993,
189.130687363480,
188.732908862760,
188.267834735326,
187.963377794302,
188.102105312079,
188.412882836827,
188.054840358786,
187.485523074778,
186.977738532323,
186.640145001627,
186.482455732677,
187.696983780267,
188.643584142771,
189.248222335827,
189.528791188363,
189.597109262444,
190.383696611981,
190.846307570758,
191.589766231352,
192.239159734164,
192.725844680950,
192.708230701306,
192.645814937027,
193.000883022726,
193.829437189199,
195.657480131989,
198.055072733188,
200.282520797509,
200.063902960450,
199.864758098248,
199.249058883673,
196.060277919784,
195.014407212901,
194.614862666729,
194.516614769717,
194.341869219687,
194.671747920249,
196.591485801924,
197.037505228424,
196.998931077752,
197.818050843519,
197.737974624715,
197.749546869917,
196.618301807873,
196.390388994748,
196.437049774597,
196.086164428127,
195.993493516754,
195.944323093368,
196.115118278570,
196.566249941906,
196.530650183576,
197.460937863085,
198.458567644188,
199.329646326161,
200.281498350142,
201.595296742111,
202.557930938328,
203.512617930009,
204.115257703211,
204.768926895013,
204.635683413115,
204.195612771297,
204.097876097969,
204.042478040619,
203.398010875122,
202.720360645071,
203.926755588604,
204.995910210531,
205.186829018915,
206.133475856300,
206.017660454524,
205.640237951387,
205.498628991030,
205.744667007482,
205.839940512153,
205.213273225821,
204.610447553098,
204.704187386717,
205.021982618395,
204.929822930706,
204.586698889250,
204.425198680113,
204.407724125110,
204.090440117117,
203.957940233304,
204.051494167403,
203.920342055119,
203.940233303899]
a = signal.detrend(z)
print(a)
