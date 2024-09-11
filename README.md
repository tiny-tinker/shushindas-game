# Shushinda's Game


<p align="center">
    <img src="./images/shushinda_at_desk_1.jpeg" width="300">
</p>

Shushinda Hushwhisper, born into a long line of unremarkable wizards, held no extraordinary promise. In fact, her talent lay in making the ordinary a touch chaotic. A magical accident quite early in her studies left her spells prone to targeting library materials rather than their intended subjects. This penchant for rearranging spellbooks and summoning dusty tomes was a source of constant frustration for her professors, but a secret delight for Shushinda.
Under the guise of exasperated sighs and mutterings of incompetence, a mischievous grin would frequently tug at her lips. A flick of her wand could send an entire shelf of grimoires waltzing across the room, or transform a stern treatise on the 'Dangers of Spontaneous Polymorphism' into a flock of startled pigeons. The whispers of her name down the hushed corridors of Unseen University were both a warning and a promise – Shushinda Hushwhisper was in the vicinity, and misplaced manuscripts were sure to follow.

# Instrumentation
This application is instrumented with [Weave](https://weave-docs.wandb.ai/). When running, interactions with Shushinda are traced using the `@weave.op` decorator for example the `LanguageModel.predict` function:

```
    @weave.op
    def predict(self, question: str, context: str = None):
        """Predict the response based on the input question and context.

        Args:
            question (str): The input question.
            context (str, optional): Additional context for the model.

        Returns:
            dict: Response containing the model's output and call ID.
        """
```

Which is traced in the Weights & Biases UI:

<kbd> ![W and B UI](./images/predict_trace.png) </kbd>


Feedback is captured and logged to the relevant trace:

<kbd> ![feedback](./images/feedback.png) </kbd>


Different LLMs can be used by changing the dropdown and the model versions (and prompts!) are tracked in Weave:

<kbd> ![models](./images/models.png) </kbd>


Finally, evaluations can be performed to detail and track how well Shushinda's responses adhere to her personality. This is a rather complex evaluation to show the versatility of custom scorers. 

<kbd> ![comparing evaluations](./images/compare_evals.png) </kbd>


# Installation

Install the spaCy libraries

```bash
python -m spacy download en_core_web_md
```


// TODO: Add more vector databases
If you're using BigQuery, this will create the dataset and tables to store the embeddings. 

## Google Family
```bash
export MY_PROJECT=`gcloud config get-value project`
echo "Using GCP project '$MY_PROJECT'"...

export REGION="us-central1"
export BQ_REGION="US"
export EMB_MODEL="textembedding-gecko@002"


export TF_VAR_project=$MY_PROJECT
export TF_VAR_bq_region=$BQ_REGION
```

TODO: Sort out what to do about this. BQ is not really necessary. 

Now run some terraform.

```bash
cd ./terraform
terraform init
terraform plan
terraform apply
```

There is not a terraform module for BigQuery models.

```bash
read -r -d '' QUERY <<-EOQ
   CREATE OR REPLACE MODEL shushindas_stuff.embedding_model
   REMOTE WITH CONNECTION \`us.vertex_ai\`
   OPTIONS (ENDPOINT = '${EMB_MODEL}')
EOQ

bq query --use_legacy_sql=false $QUERY

```


## OpenAI
Set `OPENAI_API_KEY` with a valid OpenAI key.

## Pinecone
Set `PINECONE_KEY` with a valid Pinecone key.


# Architecture



# Backstory

**Important facts about Shushinda Hushwisper**

* Shushinda Hushwhisper, despite her chaotic magic, managed to graduate from Unseen University – much to the surprise of her professors.
* She has a peculiar fondness for rearranging library stacks, especially when she's bored or startled.
* No one can make a stern treatise on the 'Perils of Thaumaturgy' dance quite like Shushinda.
* While her spells rarely hit their intended target, the resulting effects are often far more entertaining.
* Shushinda believes that 'a little chaos never hurt anyone' – a sentiment not shared by the librarians of Unseen University.


# Useful Notes and Scripts


## Searching the embeddings
```sql
-- Generate search vectors to search chunks.chunk_vector
DECLARE test_search STRING;

SET test_search = """
Does Shushinda like chaos?
""";

with stuff as (SELECT ml_generate_embedding_result as embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL shushindas_stuff.embedding_model,
    (SELECT test_search AS content),
    STRUCT('RETRIEVAL_QUERY' AS task_type)
))

SELECT base.id, base.doc_name, base.chunk_text, distance 
  FROM VECTOR_SEARCH(
    TABLE shushindas_stuff.chunks, 
    'chunk_vector',
    (select embedding from stuff),
    'embedding',
    top_k => 2,
    distance_type => 'EUCLIDEAN' -- change to COSINE or EUCLIDEAN
    )
ORDER BY distance ASC;
```


[This repo](https://github.com/GoogleCloudPlatform/bigquery-ml-utils/blob/master/notebooks/bqml-inference-remote-model-tutorial.md) is super helpful.
```python

the_embeddings = ingest.get_embeddings( "Does Shushinda like chaos?" )
# the_embeddings = [0.02221396565437317, 0.00014657355495728552, -0.024489620700478554, 0.011608565226197243, -0.002929019508883357, 0.02924383617937565, -0.02046835608780384, 0.016885455697774887, -0.00864826887845993, 0.012288357131183147, 0.012555575929582119, 0.001374410348944366, 0.012916827574372292, -0.044640734791755676, -0.054656244814395905, -0.019695278257131577, -0.021653011441230774, 0.02657560631632805, 0.016246706247329712, -0.0027190824039280415, -0.08802277594804764, 0.01858498901128769, -0.01975671574473381, 0.026148419827222824, -0.015504020266234875, -0.0944262370467186, 0.005782311782240868, 0.025553785264492035, -0.03261187672615051, -0.025473080575466156, 0.0010702598374336958, 0.02094910852611065, -0.011228285729885101, -0.020402580499649048, 0.012280784547328949, 0.029270142316818237, -0.01313091441988945, 0.0006272306200116873, -0.031795602291822433, 0.014006344601511955, 0.024740643799304962, -0.04309837147593498, 0.015480653382837772, 0.020655935630202293, -0.00560109643265605, -0.03875568136572838, -0.04001442342996597, 0.009688551537692547, -0.027673324570059776, -0.08114978671073914, 0.0012176957679912448, -0.046539805829524994, -0.043710775673389435, 0.07622764259576797, -0.006444492377340794, 0.0072199031710624695, -0.06275494396686554, -0.01781676523387432, -0.014350585639476776, -0.030604850500822067, 0.011877934448421001, -0.012906053103506565, -0.041187383234500885, -0.06819437444210052, -0.049748633056879044, 0.030264422297477722, 0.06699278205633163, -0.010078245773911476, -0.014543038792908192, -0.04645238444209099, 0.047084711492061615, 0.022130850702524185, 0.019711008295416832, -0.012511317618191242, 0.03130011633038521, 0.005294014234095812, 0.00032706715865060687, 0.04408246651291847, 0.04698110744357109, -0.045033201575279236, -0.061295729130506516, -0.06430716812610626, 0.025999821722507477, -0.11375676095485687, -0.006040435284376144, 0.021044671535491943, -0.0013530892319977283, 0.023475281894207, 0.005942022893577814, 0.03633888438344002, -0.03534093499183655, 0.004524814896285534, 0.01703844591975212, 0.054466813802719116, -0.01995375193655491, -0.027257589623332024, -0.0090001430362463, -0.04010026529431343, 0.02570676989853382, -0.047491855919361115, -0.03908655792474747, -0.04466690495610237, 0.023335682228207588, 0.011043029837310314, 0.002957199700176716, 0.02946447767317295, -0.031232431530952454, 0.08193118870258331, -0.03549640253186226, -0.049940675497055054, 0.0032532312907278538, 0.0038889225106686354, -0.026838257908821106, 0.04296498000621796, 0.009934239089488983, -0.058551136404275894, 0.0004897331236861646, -0.04017578065395355, 0.052970532327890396, 0.047579456120729446, -0.01793588511645794, -0.02097506634891033, 0.012991413474082947, 0.014137662015855312, -0.010885996744036674, 0.00775905093178153, 0.03178679198026657, 0.01583774760365486, -0.020001254975795746, -0.031189674511551857, 0.023847753182053566, -0.001134884194470942, 0.0630379244685173, -0.017548691481351852, 0.01636488363146782, 0.0043062446638941765, 0.020822547376155853, -0.025698870420455933, -0.0018651779973879457, 0.03308401629328728, -0.08559002727270126, 0.03364025428891182, -0.007478957064449787, -0.008548377081751823, 0.008949530310928822, 0.009636789560317993, -0.018287142738699913, -0.02270699106156826, -0.03361350670456886, -0.03644045814871788, 0.02381129190325737, -0.04372499883174896, -0.00043689284939318895, 0.021298514679074287, 0.013872882351279259, 0.004740679170936346, 0.0027382508851587772, 0.038777247071266174, -0.019750941544771194, -0.051946137100458145, 0.03795715421438217, -0.03310376778244972, -0.03844566270709038, -0.008380144834518433, 0.04155857115983963, 0.07344869524240494, -0.007253425661474466, 0.025118451565504074, 0.009000027552247047, -0.013680622912943363, 0.042468730360269547, -0.13106004893779755, 0.059109628200531006, -0.04883896932005882, 0.04747910425066948, 0.06441211700439453, -0.029951775446534157, 0.04436487704515457, -0.013063841499388218, -0.023434633389115334, -0.010888051241636276, -0.026144644245505333, 0.0022322488948702812, 0.020343221724033356, -0.059592969715595245, -0.0014499943936243653, -0.02854619361460209, -0.040795836597681046, -0.006362221669405699, 0.03377142548561096, -0.01776379533112049, 0.00486114202067256, 0.028908004984259605, -0.06738484650850296, 0.01650341972708702, -0.006848697084933519, 0.06958719342947006, -0.08919025212526321, -0.04224490001797676, 0.04502376168966293, 0.00429100077599287, 0.002857898361980915, 0.03283641114830971, 0.011818353086709976, -0.017341701313853264, -0.005515646189451218, -0.04310605674982071, -0.011285388842225075, 0.00041626166785135865, 0.004817735403776169, -0.0013229966862127185, 0.013408200815320015, 0.05246807262301445, 0.0005852207541465759, -0.03039819560945034, 0.02382013387978077, 0.004589597228914499, -0.029937952756881714, -0.04465390369296074, 0.01272740587592125, -0.01804579235613346, 0.004984810017049313, -0.013367055915296078, 0.024594375863671303, 0.017489902675151825, 0.05506369471549988, -0.009842471219599247, 0.0010264713782817125, 0.054475635290145874, 0.0003988079843111336, -0.01011872012168169, 0.0068227313458919525, 0.034129150211811066, 0.009275369346141815, 0.004934951663017273, 0.048619594424963, 0.0064437417313456535, 0.02788860909640789, -0.05220768228173256, 0.03474871814250946, 0.016685806214809418, 0.07874203473329544, -0.058755211532115936, -0.039826009422540665, -0.023202385753393173, 0.032182030379772186, -0.02607368677854538, 0.023118458688259125, 0.018666205927729607, -0.0264764241874218, 0.017963387072086334, -0.045363932847976685, -0.0055824038572609425, 0.03456447646021843, 0.027875393629074097, -0.007407888770103455, -0.009009672328829765, 0.026524461805820465, -0.026044661179184914, -0.013166236691176891, -0.019191663712263107, -0.030465103685855865, 0.029757380485534668, 0.16178859770298004, -0.06622239947319031, -0.021746600046753883, -0.07911044359207153, -0.0009943479672074318, 0.03598283603787422, -0.049128204584121704, -0.014760232530534267, -0.013245361857116222, 0.08275444805622101, 0.06413588672876358, 0.00861996691673994, 0.019085867330431938, -0.03732171282172203, 0.054721444845199585, 0.014663816429674625, 0.09789592772722244, 0.021758878603577614, -0.008020483888685703, -0.01029240433126688, 0.04734975844621658, -0.0022347583435475826, 0.056138549000024796, 0.004358310718089342, 0.02772676572203636, -0.08093529939651489, -0.013139128684997559, 0.009070737287402153, 0.0010475425515323877, -0.0005082953139208257, 0.06703340262174606, 0.009573453105986118, 0.01861725002527237, 0.054830487817525864, -0.019099287688732147, -0.014797779731452465, 0.036480408161878586, 0.013844831846654415, -0.04967723786830902, -0.008897935971617699, -0.028170717880129814, -0.06453864276409149, -0.032016389071941376, 0.024518845602869987, -0.005310502834618092, 0.06683289259672165, 0.003849174128845334, 0.041090551763772964, -0.04271543398499489, 0.008504665456712246, -0.0030985339544713497, -0.025119904428720474, 0.047567497938871384, 0.022066544741392136, -0.0162106491625309, -0.030884001404047012, 0.038564857095479965, -0.057813603430986404, -0.02264447882771492, 0.051309000700712204, -0.00840203557163477, 0.042157504707574844, -0.01930125616490841, -0.006746223196387291, -0.016412721946835518, 0.0642205998301506, -0.04916603863239288, 0.004490646068006754, 0.04536733031272888, -0.017723288387060165, 0.03162682056427002, 0.03987530991435051, -0.0023837455082684755, -0.00048097685794346035, 0.0736129954457283, -0.00842868909239769, -0.07772421836853027, 0.0023162264842540026, -0.046766497194767, -0.019283287227153778, 0.008551754988729954, -0.05680389702320099, -0.01014444325119257, -0.07101256400346756, -0.009744806215167046, -0.04654211550951004, -0.03791253641247749, 0.017938796430826187, 0.029148567467927933, -0.03811148554086685, 0.0439155176281929, 0.007877965457737446, 0.015416344627737999, 0.03305092826485634, -0.031680043786764145, 0.0022651059553027153, 0.030278511345386505, 0.021518947556614876, -0.014078435488045216, -0.016776468604803085, 0.007888839580118656, -0.0025104512460529804, 0.04691428318619728, -0.03191465139389038, 0.017738956958055496, -0.012868190184235573, 0.0033205400686711073, -0.01949278451502323, 0.013136064633727074, -0.033476147800683975, -0.03415351361036301, -0.03940068930387497, -0.005229434929788113, -0.04263743385672569, 0.02784830518066883, 0.003910962492227554, 0.001823602942749858, -0.038988541811704636, 0.005696703679859638, 0.024536537006497383, 0.002885945374146104, 0.010168274864554405, 0.022547194734215736, 0.030367955565452576, 0.028471995145082474, -0.003946809563785791, 0.005905213765799999, 0.018948107957839966, 0.05387052148580551, 0.010322538204491138, -0.03857804834842682, 0.020985420793294907, 0.0738590806722641, -0.046963199973106384, -0.007428914774209261, -0.047434501349925995, -0.05741937831044197, -0.011308429762721062, 0.018489675596356392, -0.026110054925084114, -0.080315962433815, -0.006273708771914244, -0.031470686197280884, 0.0015115983551368117, 0.02247951552271843, 0.04526830464601517, -0.02512129209935665, 0.005804834887385368, -0.07485983520746231, 0.06676919013261795, 0.021011311560869217, 0.042939480394124985, -0.055945899337530136, 0.0196185614913702, -0.05531685799360275, -0.03430674970149994, -0.002968382788822055, 0.005802576430141926, -0.03909386321902275, -0.08220937848091125, -0.015006528235971928, -0.018050722777843475, -0.020541230216622353, -0.03429292514920235, 0.05380875617265701, -0.00023499017697758973, -0.04382958263158798, -0.02900518663227558, -0.020909439772367477, -0.013498704880475998, -0.04467124864459038, -0.010905994102358818, -0.020701304078102112, 0.04570262134075165, 0.08022482693195343, -0.0038909311406314373, 0.010416836477816105, -0.011868271045386791, 0.01944475620985031, -0.027776699513196945, 0.015644477680325508, -0.018042953684926033, -0.015921231359243393, -0.019590750336647034, 0.020221523940563202, -0.0915428102016449, -0.01398911327123642, -0.028871264308691025, 0.01348469965159893, -0.008102206513285637, 0.11570470780134201, 0.00600392883643508, -0.019437497481703758, -0.01162784080952406, -0.04496823996305466, -0.01822674088180065, -0.012228989973664284, -0.00694819213822484, 0.013632789254188538, -0.04233609139919281, 0.07873008400201797, -0.02092193253338337, -0.030367951840162277, 0.023403706029057503, 0.05414586514234543, -0.041740819811820984, 0.0016176133649423718, 0.0017315816367045045, 0.05350998789072037, -0.0020274550188332796, -0.020617589354515076, 0.017061635851860046, 0.0019095165189355612, 0.01323896273970604, 0.029809623956680298, -0.0641903281211853, 0.021607395261526108, -0.06125848367810249, -0.056607697159051895, -0.04562138020992279, -0.005666960030794144, 0.039363205432891846, 0.012979439459741116, -0.006592575926333666, -0.027191322296857834, -0.0004665954620577395, 0.0398373119533062, -0.028114255517721176, -0.014263756573200226, 0.041356347501277924, -0.017778649926185608, 0.00746037345379591, -0.05790935084223747, 0.0005193562828935683, -0.0015492899110540748, -0.01144216489046812, -0.04993057623505592, -0.0016919750487431884, 0.0636688694357872, 0.06166733801364899, 0.008963081985712051, -0.028991321101784706, -0.005171877332031727, 0.005800161510705948, -0.025325851514935493, -0.08646659553050995, 0.0747448056936264, -0.01881168968975544, 0.018915127962827682, -0.01596623659133911, -0.01978493109345436, -0.0007991023012436926, -0.06623789668083191, -0.023521866649389267, 0.04878486692905426, 0.032666511833667755, 0.029127856716513634, -0.005433240439742804, -0.05381687358021736, 0.005941740237176418, 6.468528226832859e-06, 0.05039946362376213, 0.07070593535900116, 0.03483109176158905, -0.0525214821100235, -0.07247194647789001, 0.01203402504324913, 0.008563665673136711, 0.08006425946950912, -0.0042333039455115795, 0.03204955533146858, -0.02993682585656643, -0.04071550816297531, 0.03184147924184799, 0.007784342858940363, 0.04159489646553993, 0.0592159628868103, 0.049064747989177704, -0.06742498278617859, 0.03843722864985466, -0.029129140079021454, 0.049658916890621185, 0.006230181083083153, 0.013207037933170795, 0.03360407054424286, 0.01132137794047594, 0.0038458264898508787, 0.02751358225941658, -0.007517436519265175, -0.03731999173760414, 0.05959383770823479, 0.01990264654159546, -0.02988697588443756, 0.022991729900240898, -0.04814601317048073, -0.04248811677098274, -0.005921665579080582, 0.0162526722997427, 0.0007020576158538461, -0.024517197161912918, -0.06465976685285568, 0.033765003085136414, -0.03492804616689682, 0.025611121207475662, -0.015855589881539345, 0.006996905896812677, 0.05208852142095566, -0.051580771803855896, -0.05835044011473656, -0.020951714366674423, 0.0026810450945049524, -0.031560398638248444, -0.013237999752163887, -0.004981639329344034, 0.0067750574089586735, -0.008322712033987045, 0.04280805587768555, -0.03521089628338814, -0.012928714044392109, -0.012374991551041603, 0.06540123373270035, -0.05088198557496071, 0.0025554539170116186, 0.000540844805072993, -0.04677649959921837, 0.001682379050180316, -0.04869465157389641, -0.017107130959630013, -0.06147409602999687, 0.001827583066187799, 0.037172481417655945, -0.06439018249511719, -0.03129870817065239, -0.005100752227008343, 0.022750964388251305, -0.008904501795768738, 0.025135239586234093, -0.004423700738698244, -0.02581104077398777, 0.05454317852854729, -0.04312260076403618, -0.016802873462438583, -0.01663975603878498, -0.00041106206481345, 0.06170355901122093, -0.0091589679941535, 0.01461734063923359, 0.08269476890563965, 0.04947538301348686, 0.04036976397037506, 0.004465695470571518, 0.011827141977846622, -0.013055049814283848, -0.029604272916913033, 0.022856436669826508, 0.043877437710762024, -0.05744845047593117, 0.04131193831562996, -0.02942889928817749, -0.02513434924185276, -0.0883537009358406, -0.00861220434308052, -0.020546292886137962, 0.031139329075813293, 0.03975619003176689, -0.010609135963022709, -0.011963840574026108, 0.02972676232457161, -0.04977421835064888, -0.01223272830247879, -0.03092018887400627, 0.029746845364570618, -0.004692466929554939, -0.05351349338889122, -0.036820560693740845, 0.02842872403562069, 0.043060608208179474, 0.029459618031978607, -0.014373385347425938, -0.014723356813192368, -0.013964890502393246, 0.022778645157814026, -0.03723173961043358, 0.018069617450237274, -0.026506515219807625, -0.02593000791966915, -0.006414102856069803, -0.013422435149550438, -0.03712526708841324, -0.050251055508852005, -0.03892217203974724, 0.01571395806968212, 0.02780218794941902, -0.021419266238808632, 0.028200656175613403, -0.053896430879831314, 0.02794404700398445, 0.05651822313666344, 0.03081103041768074, 0.0062004029750823975, 0.011857961304485798, -0.07328099012374878, 0.08349525928497314, -0.013059605844318867, 0.019632283598184586, 0.004428253509104252, 0.03064955584704876, 0.0055759563110768795, -0.049496568739414215, 0.025823285803198814, -0.037140097469091415, -0.03998422995209694, 0.016970211640000343, 0.03283414617180824, 0.02163081429898739, -0.006727566476911306, -0.018842773512005806, -0.03972407430410385, 0.022173555567860603, -0.013717154040932655, 0.0008460551034659147, 0.04587830975651741, -0.014011921361088753, 0.005026585888117552, 0.010485918261110783, 0.0016253471840173006, -0.041558992117643356, -0.01741521805524826, 0.03969558700919151, -0.009914467111229897, -0.0036539272405207157, -0.020114997401833534, -0.04935021325945854, -0.03645303472876549, 0.06672517210245132, 0.0493185929954052, 0.0017104686703532934, -0.03839213401079178, 0.009268111549317837, 0.06743285804986954, 0.048806726932525635, 0.051983803510665894, -0.0449594110250473, -0.04909558221697807, 0.02601616270840168, 0.014349382370710373, -0.02206067182123661, -0.04063582792878151, -0.0169663205742836, 0.051807135343551636, 0.013408200815320015, -0.02522539161145687, 0.030568119138479233, -0.04284431412816048, 0.010228876955807209, 0.02923310548067093, -0.030132489278912544, -0.026486974209547043, 0.03106025978922844, 0.03199522942304611, -0.050016943365335464, -0.04964752122759819, 0.01211269199848175, 0.03907398134469986, -0.04339764639735222, -0.033798106014728546, 0.0008739590412005782, -0.001960202818736434, 0.010329850018024445, 0.02864016219973564, 0.07716947048902512, 0.048065584152936935, -0.04939568415284157, 0.03373179957270622, 0.017062123864889145, 0.028112106025218964, -0.08533339202404022, -0.013142767362296581, 0.0021270171273499727, -0.0009432625374756753, 0.007404538802802563, 0.054850541055202484, -0.005165720358490944, 0.0298172440379858, 0.0026527734007686377, 0.003676073392853141, -0.03686532378196716, -0.05778361111879349, 0.023261351510882378, 0.007837084122002125, -0.032904624938964844, -0.023400364443659782, 0.02931959182024002, -0.01630706898868084, 0.0004488704726099968, -0.04311950504779816, 0.05758295953273773, -0.01760989986360073, -0.004865635186433792, -0.06297292560338974, 0.0221310593187809, 0.02920559234917164, 0.02918471209704876, 0.031172743067145348, -0.0662895068526268]
query = f"""
SELECT base.id, base.doc_name, base.chunk_text, distance 
  FROM VECTOR_SEARCH(
    TABLE sushindas_stuff.chunks, 
    'chunk_vector',
    (select @search_embedding as embedding),
    'embedding',
    top_k => 2,
    distance_type => 'EUCLIDEAN' -- change to COSINE or EUCLIDEAN
    )
ORDER BY distance ASC;
"""

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("search_embedding", "FLOAT", the_embeddings),
    ]
)
query_job = client.query(query, job_config=job_config)  # Make an API request.

for row in query_job:
    print( f"id: {row.id}, doc_name: {row.doc_name}, chunk_text: {row.chunk_text}, distance: {row.distance}")
```