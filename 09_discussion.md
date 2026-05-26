---
title: Discussion
numbering:
  enumerator: 0.%s
---

Every academic discipline has its own ways of deciding what counts as valid knowledge (see [epistemology](<wiki:Epistemology>)).
If biomedical researchers endorse empiricism and objectivity, then why are representative images sufficient to make claims?
A representative image shows that an observation is possible, but a single image cannot describe frequency or variance.
The scientific method demands reproducibility, yet representative images offer no way to communicate repeatability without quantitative analysis.
In practice, representative images can show rare, idealized, or exaggerated examples to embellish results.
The literature is full of exciting outcomes, but at the bench, experiments are nuanced and prone to failure.

Representative images are figures, which should not be confused with data.
Data is the unprocessed file from the microscope, while figures are visual representations of data, shaped by the microscopist.
Figures are the interface for peer review between the experimentalist and coauthors, principal investigators, reviewers and readers.
Indeed, peer review seldom involves data and scientific discourse is limited by the information provided in figures.
This bottleneck is alleviated by virtual microscopy, which is practical to the broader bioimaging community.
For example, OME-Zarr [@doi:10.1007/s00418-023-02209-1] supports several modalities including multiplexed imaging, spatial transcriptomics, light-sheet, infrared, scanning EM and digital pathology.
To maximize accessibility for readers, embedded interactive viewers retrieve data from cloud storage, which is supported by this journal.
Virtual microscopy requires an internet connection, but this limitation is relatively unimportant for contemporary and future publishing.
Web-based microscopy maps are trivial to make and share on Github [@doi:10.1242/jcs.262198], so they can be readily requested by reviewers of all journals.
Data transparency could incentivize publishers to adopt this technology as an alternative to static figures.
Because it nurtures open and rigorous peer review, microscopists and principal investigators might find virtual microscopy advantageous to make more robust claims with their bioimaging data.

Barkley advocated for the use of virtual microscopy as an "unbiased" method to replicate highly magnified representative images.
This process involves automated sampling of reasonable populations from which the original claim should be reliably observed.
Virtual microscopy is "unbiased" to the extent that the data was sampled with robotics and is hosted interactively on the web.
To test this approach, an immunofluorescence imaging experiment in reovirus-infected cells [@doi:10.1128/mBio.01253-18] was reexamined with virtual microscopy.
Barkley hypothesized that fluorescent proteins could have restructured the ER [@doi:10.1083/jcb.200306020], which was not addressed by the previous investigators.
In replication attempts, ER remodeling was observed in transfected cells expressing mCherry-KDEL in the absence of reovirus.
This suggests that artifacts from mCherry-ER reporters could have been misinterpreted as an outcome of infection.
Otherwise, it was not clear that reovirus affected ER morphology, certainly not to the extent of fragmentation, collapse, and aggregation as described, though we recognize that this interpretation was influenced by confirmation bias.

When Barkley sought to uncover ground truths through virtual microscopy, Parker offered the following perspective: 
There is no ground truth. 
The term is misleading, as all "established facts" are simply our current interpretation of available findings, data, or measurements. 
Science does not generate truth; it generates findings or data that are analyzed and then interpreted by humans, and those interpretations can differ, sometimes widely. 
Through the accrual of many findings, a consensus picture of current "reality" emerges that is sometimes called "ground truth," but this is only a snapshot, not the final word.
Truth is not a straightforward concept. 
All truths can be challenged, and while some become more certain with the passage of time, they are never without caveats. 
Science is the process of observing, imagining explanations for the observations, turning those explanations into hypotheses, and then testing them through experimentation. 
The truths we derive from this process are "interpreted truths" that may become irrelevant as new information is gathered. 
Our job as scientists is not to identify ground truths, but to test hypotheses and garner evidence that supports or contradicts them, and to challenge the prevailing consensus.
