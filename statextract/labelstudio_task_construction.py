
import json
from pathlib import Path
import shutil

import tqdm

from statextract.helpers import add_image_ruler_overlay, collect_model_inputs, collect_model_text, fetch_work, form_path_base, md_image_paths
from statextract.md_retriever import parse_work
from statextract.typedefs import PaperMD

    #     with open("short_mds_ids.json") as f:
    #         ids = json.load(f)

    #     print(ids)

    #     result_dir = Path("results/claude-3-5-sonnet-latest")
    #     result_dir.mkdir(parents=True, exist_ok=True)

    #     import tqdm

    #     for id in tqdm.tqdm(ids):
    #         path = Path(result_dir) / f"{id.split('/')[-1]}.json"
    #         if path.exists():
    #             continue
    #         work = fetch_work(id)
    #         md = parse_work(work)

from PIL import Image, ImageDraw, ImageFont
from label_studio_sdk import Client
import argparse

args = argparse.ArgumentParser()
args.add_argument("--output_workdir", type=str, default="/tmp/labelstudio_output")
args.add_argument("--ssh_key_path", type=str)
LABEL_STUDIO_URL = 'pcurves.zygi.me'
LABEL_STUDIO_URL_WITH_SCHEME = f"https://{LABEL_STUDIO_URL}"

def work(args: argparse.Namespace):
    
    API_KEY = 'abeafeb1a1484ed87e0eb6c5f3e61b15b859935a'

    ls = Client(url=LABEL_STUDIO_URL_WITH_SCHEME, api_key=API_KEY)
    ls.check_connection()
    
    proj = ls.get_project(id=1)

    
    with open("short_mds_ids.json") as f:
        ids = json.load(f)


    files = [parse_work(fetch_work(id)) for id in tqdm.tqdm(ids)]



    # def collect_model_images(md: PaperMD, image_path: Path = Path("data/images")):
    #     image_files = md_image_paths(md, image_path)
        # images_transformed = [add_image_ruler_overlay(Image.open(img)) for img in image_files]
        # image_data = [(base64.b64encode(img_to_bytes(img)).decode(), img.size) for img in images_transformed]
    # return image_data
    
    pdf_paths = [(Path("data/pdfs") / (form_path_base(md) + ".pdf")).resolve().absolute() for md in files]
    
    dest_root = Path(args.output_workdir)
    pdf_dest = dest_root / "pdfs"
    pdf_dest.mkdir(parents=True, exist_ok=True)
 
    # create pdf symlinks
    for pdf_path in tqdm.tqdm(pdf_paths):
        if (pdf_dest / pdf_path.name).exists(follow_symlinks=False):
            continue
        (pdf_dest / pdf_path.name).symlink_to(pdf_path.resolve())
    
    # now, images
    image_dest = dest_root / "images"
    image_dest.mkdir(parents=True, exist_ok=True)
    for md in tqdm.tqdm(files):
        for img_path in md_image_paths(md, Path("data/images")):
            if (image_dest / img_path.name).exists(follow_symlinks=False):
                continue
            (image_dest / img_path.name).symlink_to(img_path.resolve())
    
    # now, rsync
    ssh_key_path = Path(args.ssh_key_path)
    if not ssh_key_path.exists():
        raise ValueError(f"SSH key path {ssh_key_path} does not exist")
    
    cmd = f"""rsync -avzL -e "ssh -i {str(ssh_key_path)}" {str(dest_root)}/* root@{LABEL_STUDIO_URL}:/root/pcurves/ls_files/pcurves_dataset"""
    print(cmd)
    
    # forward stdout and stderr to shell
    import subprocess
    subprocess.run(cmd, shell=True)
    
    container_relative_root = f"{LABEL_STUDIO_URL_WITH_SCHEME}/data/local-files/?d=pcurves_dataset/"
    
    # generate tasks
    def convert_task(md: PaperMD, pdf_path: Path, image_paths: list[Path]):
        full_pdf_path = str(container_relative_root + "/pdfs/" + pdf_path.name)
        obj = {
            "authors": md.author_names,
            "title": md.title,
            "id": md.id,
            "pdf_path": full_pdf_path,
            "pdf_link_tag": f""" <a href="{full_pdf_path}"> {md.title} </a>""",
            "image_paths": [str(container_relative_root + "/images/" + p.name) for p in image_paths],
            "num_images": len(image_paths),
        }
        return obj
    
    # write out json
    task_objs = [convert_task(md, pdf_path, md_image_paths(md, Path("data/images"))) for md, pdf_path in tqdm.tqdm(zip(files, pdf_paths))]
    with open("tasks.json", "w") as f:
        json.dump(task_objs, f)
    
    proj.delete_all_tasks()
    
    # upload tasks
    proj.import_tasks(task_objs)
    
    
    # delete all
    
    
    # for md in files:
    #     text = collect_model_text(md)
    #     print(text[:1000])
    #     print(images)
    #     break
    
    # text, images = collect_model_inputs(md)

    # for id in tqdm.tqdm(ids):
        # path = Path(result_dir) / f"{id.split('/')[-1]}.json"
        # if path.exists():
        #     continue
        # work = fetch_work(id)
        # md = parse_work(work)
        
        
        
def template_generator():
    def header():
        return """
    <View display="block">  
        <Header value="Paper info:" size="2"/>
        <Header value="Title: $title" size="3"/>
        <Header value="Authors: $authors" size="4"/>
    </View>
        """
    
    
    
    def base(claim_num: int, include_valid: bool = True, hidden_unless_more_claims: bool = False):
        
        valid_part = f"""
<Header value="First, evaluate the paper as a whole. Can p-values be extracted from it? If so, select 'Valid'. If not, select the most appropriate reason below." size="4" style="font-size: 1.2em"/>
<Choices name="valid" choice="single-radio" toName="pages{claim_num}" required="true">
<Choice value="Valid" alias="valid" selected="false"/>
<Choice value="Invalid - no p-values, only qualitative" selected="false"/>
<Choice value="Invalid - not in English" selected="false"/>
<Choice value="Invalid - wrong paper" selected="false"/>
<Choice value="Invalid - malformatted" selected="false"/>
<Choice value="Invalid - other" selected="false"/>
</Choices>
<View visibleWhen="choice-selected" whenTagName="valid" whenChoiceValue="Invalid - other">
    <TextArea name="invalid-details" placeholder="Brief reason for being invalid..."
                toName="pages{claim_num}" 
            showSubmitButton="false" rows="4" editable="true" maxSubmissions="1"/>
</View>


"""         
        valid_view_filter = ' visibleWhen="choice-selected" whenTagName="valid" whenChoiceValue="Valid"'


        hidden_unless_more_claims_filter = f' visibleWhen="choice-selected" whenTagName="claim{claim_num-1}-has-next-claim" whenChoiceValue="Yes"'
        
        
        info_header = f"""
    <Header value="Claim {claim_num}" size="2"/>
""" if claim_num == 1 else f'<Header value="Claim {claim_num}" size="2"/>'
    
        next_claim_question = f"""
        
        <Header value="Is there another, substantially different claim in the abstract?"/>
        <Choices name="claim{claim_num}-has-next-claim" choice="single-radio" toName="pages{claim_num}">
        <Choice value="Yes" />
        <Choice value="No" />
        </Choices>
        """
    
        CLAIM_DETAILS_PLACEHOLDER = """\
If the claim above is ambiguous and doesn't identify a precise single hypothesis, please add details to make the claim specific, so that the details unambiguously identify \
a p-value reported in the paper. Make an arbitrary choice if necessary. E.g. if the claim was 'The intervention significantly changed personality scores of participants', \
and the result table reported five rows for different personality scores, please choose one score and describe the choice made.
"""

#         CLAIM_DETAILS_PLACEHOLDER = """\
# Given the claim above, choose one P-value in the paper that represents the claim. Describe any choices made, if any, \
# in this field. For example, if the claim is 'The intervention significantly changed personality scores of participants', \
# and the result table reports five rows for different personality score types, choose one type and describe the choice made. \
# A reviewer should be able to read the claim above, and the specific details here, and unambiguously identify the P-value in the paper."""

        return f"""
<View style="display: flex; flex-direction: row" {hidden_unless_more_claims_filter if hidden_unless_more_claims else ""}>

    <View style="flex-basis: 30%; border: 1px solid; padding: 5px">
    {valid_part if include_valid else ""}



    <View {valid_view_filter if include_valid else ""}>
        {info_header}
        <Header value="Claim from abstract"/>
        <TextArea name="claim{claim_num}-claim" 
                toName="pages{claim_num}" 
                placeholder="Choose one claim from the paper's abstract. Put it here, ideally copied verbatim from the abstract."
                rows="4" 
                editable="true"
                maxSubmissions="1"
                showSubmitButton="false"/>

        <Header value="Specific claim details"/>
        <TextArea name="claim{claim_num}-claim-narrowing" 
                toName="pages{claim_num}" 
                placeholder="{CLAIM_DETAILS_PLACEHOLDER}"
                rows="4" 
                editable="true"
                maxSubmissions="1"
                showSubmitButton="false"/>
        
        <Header value="P-value"/>
        <Choices name="claim{claim_num}-pvalue-type" choice="single-radio" toName="pages{claim_num}">
        <Choice value="Bound (e.g. p &lt; 0.01)" alias="bound" />
        <Choice value="Exact (p = 0.0162)" alias="exact" />
        </Choices>
        <TextArea name="claim{claim_num}-pvalue-val" toName="pages{claim_num}" placeholder="e.g. 0.05, 0.00215, 3.2e-5" editable="false" 
                showSubmitButton="false" maxSubmissions="1"/>
                
        <Header value="Other test statistics (one per entry)"/>                
        <TextArea name="claim{claim_num}-other-test-stats" toName="pages{claim_num}" placeholder="e.g. t(88)=2.1, r(147)=.246, F(1,100)=9.1, f(2,210)=4.45, Z=3.45, chi2(1)=9.1, r(77)=.47, chi2(2)=8.74" editable="false" 
                showSubmitButton="true" rows="1"/>
        
        <Header value="Mark the region where the p-value is reported in the paper's images to the right."/>        
        <Choices name="claim{claim_num}-pvalue-reported-confirmation" toName="pages{claim_num}">
        <Choice value="Done" selected="false" />
        </Choices>
                
        {next_claim_question if claim_num < 3 else ""}

    </View>
    </View>

    <View  style="flex-basis: 70%; border: 1px solid; padding: 5px; height: 100%">
        <Header value="PDF link (for copying text)"/>
        <HyperText name="pdfVal{claim_num}_ignore" value="$pdf_link_tag"/>
        <Header value="Images (for labeling)"/>
        <Rectangle name="claimSource{claim_num}" toName="pages{claim_num}" />       
        <Style>
        div[class^="container--"] {{max-height: fit-content !important;}}
        </Style> 
        <Image name="pages{claim_num}" valueList="$image_paths" zoom="true" defaultZoom="original"/>
    </View>
</View>
        
"""

    return "<View>\n" + header() + "\n" + base(1, include_valid=True, hidden_unless_more_claims=False) + "\n" + base(2, include_valid=False, hidden_unless_more_claims=True) + "\n" + base(3, include_valid=False, hidden_unless_more_claims=True) + "\n</View>"
        
if __name__ == "__main__":
    # copy to clipboard
    import pyperclip
    pyperclip.copy(template_generator())
    # work(args.parse_args())