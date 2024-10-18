import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { ClaimDetailsSuccess, DetailedClaimRequest, getWorkWorkWorkIdGet, WrappedAbort } from '../generated_types'
import { stripId } from '../utils'
import { Accordion, AccordionItem, Button, CircularProgress, Code, Modal, ModalContent, Popover, PopoverContent, PopoverTrigger, useDisclosure } from '@nextui-org/react'
import { useState } from 'react'

export const Route = createFileRoute('/work/$id')({
  component: WorkComponent,
})


function ImageWithBox(params: { src: string, bounding_box: [number, number, number, number] }) {
  // function ImageWithBox(params: {src: string, bounding_box: [number, number, number, number]}) {
  return <div style={{ position: 'relative', display: 'inline-block' }}>
    <img src={params.src} />
    <div style={{ position: 'absolute', left: params.bounding_box[0], top: params.bounding_box[1], width: params.bounding_box[2] - params.bounding_box[0], height: params.bounding_box[3] - params.bounding_box[1], border: '2px solid red', pointerEvents: 'none' }}></div>
  </div>
}

function ClaimImage(params: { claim: DetailedClaimRequest, id: string }) {
  return <ImageWithBox src={`http://localhost:8000/image/${params.id}/${params.claim.statistical_support_page_number - 1}`} bounding_box={params.claim.statistical_support_bounding_box} />;
}

function CroppedClaimImage(params: { claim: DetailedClaimRequest, id: string, padding?: number }) {
  // Instead of just showing the image, show a cropped version of the image
  // centered on the bounding box of the claim
  const [x, y, x2, y2] = params.claim.statistical_support_bounding_box;
  const padding = params.padding || 20; // Add some padding around the bounding box
  const width = x2 - x;
  const height = y2 - y;

  const style = {
    objectFit: 'none' as const,
    objectPosition: `${-x + padding}px ${-y + padding}px`,
    width: `${width + 2 * padding}px`,
    height: `${height + 2 * padding}px`,
    maxWidth: 'none',
    // border: '2px solid red',
  };

  return (
    <div style={{ overflow: 'hidden', display: 'inline-block' }}>
      <img
        src={`http://localhost:8000/image/${params.id}/${params.claim.statistical_support_page_number - 1}`}
        style={style}
        alt="Cropped claim image"
      />
    </div>
  );
}

function PValue(params: { claim: ClaimDetailsSuccess }) {
  const claim = params.claim.claim;
  if (claim.p_value_exact) {
    return <span>{`p = ${claim.p_value_exact}`}</span>;
  }
  if (claim.p_value_computed_R_script) {
    const [showScript, setShowScript] = useState(true);
    const button = <Button onClick={() => setShowScript(!showScript)}>{showScript ? 'Hide' : 'Show'} R script</Button>

    return <>
      <span>{`p = ${params.claim.pvalue}`} {button}</span>
      {showScript && <div className="p-2 bg-default-100 rounded-md"><Code><pre className="whitespace-pre-wrap">{claim.p_value_computed_R_script}</pre></Code></div>}
    </>;
  }
  if (claim.p_value_bound) {
    return <span>{`p < ${claim.p_value_bound}`}</span>;
  }
  return 'Unknown';
}

function ClaimDetails(params: { claim: ClaimDetailsSuccess, id: string }) {
  const claimBase = params.claim.claim;
  const { isOpen, onOpen, onOpenChange } = useDisclosure();


  return <div className="p-4">
    <div className="mb-4"><span className="font-bold">Chain of thought:</span> <pre className="whitespace-pre-wrap bg-default-200 p-2 rounded-md">{claimBase.chain_of_thought_md}</pre></div>
    <div className="mb-4"><span className="font-bold">Claim summary:</span> {claimBase.claim_summary}</div>
    <div className="mb-4"><span className="font-bold">Statistical support summary:</span> {claimBase.statistical_support_summary_md}</div>
    <div className="mb-4"><span className="font-bold">P-value:</span> <PValue claim={params.claim} /></div>
    {/* <div className="mb-4"><span className="font-bold">Page number:</span> {claimBase.statistical_support_page_number}</div> */}
    {/* <div className="mb-4"><span className="font-bold">Bounding box:</span> {claimBase.statistical_support_bounding_box.join(', ')}</div> */}

    <div className="mb-4"><p className="font-bold">Claim image (click to expand):</p>
      <div onClick={onOpen}><CroppedClaimImage claim={claimBase} id={params.id} /></div>
      <Modal isOpen={isOpen} onOpenChange={onOpenChange} size="4xl">
        <ModalContent>
          <ClaimImage claim={claimBase} id={params.id} />
        </ModalContent>
      </Modal>
    </div>

  </div>
}

function ClaimOrError(params: { claim: ClaimDetailsSuccess | WrappedAbort | null, id: string }) {
  if (!params.claim) {
    return <div className="text-danger-500 bg-danger-100 rounded-md p-4 m-4">Error: No claim</div>
  }
  if (params.claim.tag === 'abort') {
    return <div className="text-danger-500 bg-danger-100 rounded-md p-4 m-4">Error: {params.claim.reason}</div>
  }
  if (params.claim.tag === 'claim_details_success') {
    return <ClaimDetails claim={params.claim} id={stripId(params.id)} />
  }
  throw new Error("Unexpected claim type: " + params.claim)
}

//   if ('tag' in params.claim && params.claim.tag === 'success') {
//   return <ClaimDetails claim={params.claim} id={stripId(id)} />
// }

// function FullResponse(params: {response: FullResponse}) {


function WorkComponent() {
  const { id } = Route.useParams()
  // const id = "W2009995022"

  const query = useQuery({ queryKey: ['work', id], queryFn: ({ queryKey }) => getWorkWorkWorkIdGet({ path: { work_id: queryKey[1] } }) })
  if (query.isError) {
    return <div>Error: {query.error.message}</div>
  }
  if (query.isLoading) {
    return <div className="flex justify-center items-center h-screen"><CircularProgress /></div>
  }
  const data = query.data?.data;
  if (!data) {
    return <div>No data</div>
  }

  // box_html = f"""
  // <div style="
  //     position: absolute;
  //     left: {bounding_box[0]}px;
  //     top: {bounding_box[1]}px;
  //     width: {bounding_box[2] - bounding_box[0]}px;
  //     height: {bounding_box[3] - bounding_box[1]}px;
  //     border: 2px solid red;
  //     pointer-events: none;
  // "></div>
  // """

  // # Wrap the image and bounding box in a container
  // image_html = f"""
  // <div style="position: relative; display: inline-block;">
  //     {image_html}
  //     {box_html}
  // </div>
  // """



  const title = <h1 className="text-2xl font-bold text-default-900">{data.md.title}</h1>
  const authors = <div className="text-sm text-default-500">{data.md.author_names.join(', ')}</div>

  const response = data.full_response.inner;
  if (response.tag === "abort") {
    return <div className="text-danger-500 bg-danger-100 rounded-md p-4 m-4">Error: {response.reason || 'Unknown error'}</div>
  }

  if (response.tag !== "success") {
    throw new Error("Unexpected response tag: " + response.tag)
  }

  // let images;
  // images = response.detailed_claim_results.map((claim, idx) => {
  //   // If not list, it's an error
  //   if (!Array.isArray(claim)) {
  //     console.log(claim)
  //     return <div>Error: {claim.reason}</div>
  //   }
  //   return <ClaimImage claim={claim[0]} id={stripId(id)} />;
  // });


  // if (response.claims.claims.length === 0) {
  //   return <div>No claims</div>
  // }

  let all_claims;
  if (response.claims.claims.length > 0) {
    all_claims = <Accordion variant="splitted" selectionMode="multiple" className="mb-10" defaultExpandedKeys={response.claims.claims.map((_, idx) => idx.toString())}>
      {response.claims.claims.map((claim, idx) => {
        return <AccordionItem key={idx.toString()} title={`Claim ${idx + 1}: ${claim.summary}`} classNames={{ base: "bg-default-100" }}>
          <ClaimOrError claim={response.detailed_claim_results[idx]} id={stripId(id)} />
        </AccordionItem>
      })}
    </Accordion>
  } else {
    all_claims = <div>No claims!!!!</div>
  }

  const debug = <div className="p-10 bg-default-100 text-default-900 rounded-md"><pre className="whitespace-pre-wrap">{JSON.stringify(data.md, null, 2)}</pre><pre className="whitespace-pre-wrap">{JSON.stringify(data.full_response, null, 2)}</pre></div>




  return <div className="ml-20 mr-20 mt-20 mb-20 flex flex-col">
    {title}
    {authors}
    {all_claims}
    {debug}
  </div>
}