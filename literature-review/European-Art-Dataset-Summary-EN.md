# European Art Dataset (DEArt) - Summary

## Overview
**DEArt (Dataset of European Art)** is a specialized object detection and pose classification dataset focused on European paintings from the 12th to 18th centuries.

## Key Characteristics

### Dataset Size
- **Total Images:** ~15,200 images
- **Size Category:** 10K-100K range
- **Format:** Parquet, JPEG images
- **License:** CC-BY-NC-2.0

### Content
- **Temporal Coverage:** 12th-18th century European art
- **Geographic Scope:** European cultural heritage
- **Image Type:** ~80% non-iconic paintings
- **Annotation Type:** Bounding boxes and pose classifications

### Classes and Annotations
- **Total Classes:** 69 object classes
- **Pose Classifications:** 12 possible poses for human-like objects
- **Annotations:** 46,806+ person annotations, 11,356 tree annotations, plus 67 other cultural heritage-specific classes
- **Cultural Heritage Specific:** 50+ classes unique to historical art (angels, halos, religious symbols, mythological creatures)

## Data Sources
Images collected from multiple cultural heritage institutions:
- National Museum in Warsaw (2,030 images)
- Europeana Collection (1,991 images)
- The Art Institute of Chicago (1,237 images)
- The Metropolitan Museum of Art (1,218 images)
- Rijksmuseum (1,066 images)
- National Gallery of Art (871 images)
- 20+ other institutions

## Tasks Supported
1. **Object Detection:** Detecting cultural heritage objects in historical paintings
2. **Image Classification:** Classifying images by object categories
3. **Pose Classification:** Identifying human poses in artworks

## Dataset Structure

### Annotations Format
Two configuration options available:
- **Raw format:** Original annotation structure
- **COCO format:** Compatible with standard computer vision pipelines and Transformers models

### Key Fields
- Image metadata (width, height, source)
- Bounding boxes (xmin, ymin, xmax, ymax)
- Category IDs and labels
- Pose information for human-like objects

## Limitations for Big Data Research

### Scale Issues
1. **Small Dataset Size:** ~15K images is insufficient for testing Big Data processing frameworks
2. **Not Representative:** Does not reflect production-scale workloads typical in cloud computing scenarios
3. **Limited Diversity:** Focused exclusively on European historical art (12th-18th centuries)

### Workload Characteristics
- **No Real-world Patterns:** Lacks diverse workload patterns needed for performance evaluation
- **Non-production Scale:** Cannot assess scalability, energy consumption, or resource management at scale
- **Domain-specific:** Designed for cultural heritage applications, not general Big Data processing

### Evaluation Constraints
- **Not Industry-standard:** Not recognized as a benchmark for Big Data or cloud computing research
- **Limited Metrics:** Cannot measure realistic resource utilization, SLA violations, or operational costs
- **Missing Variability:** Lacks temporal patterns, burst workloads, or diverse data processing scenarios

## Appropriate Use Cases

### Recommended For:
- Computer vision research in digital humanities
- Object detection in historical artworks
- Cultural heritage preservation and analysis
- Art history digital tools development
- Educational applications in art recognition

### NOT Recommended For:
- Big Data processing framework evaluation
- Cloud resource optimization research
- Scalability testing for distributed systems
- Energy consumption analysis in data centers
- Production workload simulation

## Critical Assessment for Research Proposals

### Why This Dataset is Inadequate for Big Data/Cloud Research:

1. **Scale Mismatch:** 
   - Real Big Data: TBs-PBs of data
   - This dataset: <20 GB total
   
2. **Workload Patterns:**
   - Real systems: Dynamic, bursty, time-varying loads
   - This dataset: Static image collection
   
3. **Evaluation Metrics:**
   - Big Data research needs: Throughput, latency, resource utilization, cost
   - This dataset enables: Object detection accuracy only

4. **Industry Relevance:**
   - Production systems: Google Cluster Trace, Alibaba Trace, Azure Functions
   - This dataset: Historical art images

## Reference
- **Paper:** arXiv:2211.01226
- **Repository:** DOI:10.5281/zenodo.6984525
- **HuggingFace:** biglam/european_art

## Conclusion
While the European Art Dataset is a valuable resource for cultural heritage and art history applications, it is fundamentally **unsuitable for Big Data and cloud computing research**. Its small scale, static nature, and domain-specific focus make it inappropriate for evaluating distributed processing frameworks, resource optimization algorithms, or scalability studies. Researchers proposing to use this dataset for Big Data thesis work should reconsider and select industry-standard datasets that reflect real production workloads.
