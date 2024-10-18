// This file is auto-generated by @hey-api/openapi-ts

export type AuthorResponse = {
    name: string;
    id: string;
    works: Array<[
        PaperMD,
        (FullExtractionResult | null)
    ]>;
};

export type ClaimDetailsSuccess = {
    tag?: 'claim_details_success';
    claim: DetailedClaimRequest;
    pvalue: (number | null);
};

export type tag = 'claim_details_success';

/**
 * A core claim of the paper that is supported by quantitative analysis and a statistical test.
 */
export type ClaimSummary = {
    /**
     * Claim summary.
     */
    summary: string;
    /**
     * The position of the claim in the text, described verbally.
     */
    position_in_text: string;
};

/**
 * A collection of core claims of the paper that are supported by quantitative analysis and a statistical test.
 * Please only choose the most important claims of the paper. They should be 1) mentioned in the abstract, AND 2) supported by some kind of statistical test.
 *
 * Output no more than 3 claims.
 */
export type Claims = {
    /**
     * The chain of thought you can use for reasoning.
     */
    chain_of_thought: string;
    /**
     * The list of claims. No more than 3.
     */
    claims: Array<ClaimSummary>;
};

export type DetailedClaimRequest = {
    /**
     * First, you can choose to think through your answer here. You can use markdown formatting.
     */
    chain_of_thought_md: string;
    /**
     * The summary of the claim being made.
     */
    claim_summary: string;
    /**
     * The summary of the statistical tests that support the claim, as well as the key test statistics and values. Specifically mention which statistical test was used by saying 'Statistical test: <test name>'. If you can't determine the test name, output 'Statistical test: UNKNOWN'. You can use markdown formatting.
     */
    statistical_support_summary_md: string;
    /**
     * The page number of the article where the statistical support is located. This should be a number between 1 and the total number of pages (provided images) in the article.
     */
    statistical_support_page_number: number;
    /**
     * The bounding box of the statistical support in the article, contained in the page provided in `statistical_support_page_number`. The box should be a tight fitting box around the text of the claim, with 4 numbers, in the format (x1, y1, x2, y2). The box should be in the coordinate system of the article, with the top left corner being (0, 0) and the bottom right corner being (width, height). Use the rulers for reference. BE VERY SPECIFIC AND ONLY INCLUDE THE SUPPORTING SENTENCES, NOT HUGE BLOCKS OF TEXT.
     */
    statistical_support_bounding_box: [
        number,
        number,
        number,
        number
    ];
    /**
     * If the paper reports an exact p-value for the claim (p = <value>), output it here. Otherwise, if it reports a bound like p < <value> or simply doesn't mention it, output None.
     */
    p_value_exact: (number | null);
    /**
     * If the paper does NOT report an exact p-value, but does report test statistics that can be used to calculate it AND SPECIFICALLY MENTIONS WHICH STATISTICAL TEST WAS USED, output the calculation here. The calculation should be a valid R script that can be used to compute the p-value. It should take no external input - you should enter the numbers inline. Your script's final line should assign the p-value to a variable named `pvalue`. FOLLOW THE ARTICLE'S METHOD AS DESCRIBED AND DON'T MAKE ASSUMPTIONS. If you need to guess the statistical test, output null instead of guessing.
     */
    p_value_computed_R_script: (string | null);
    /**
     * As a last resort, if you cannot find the p-value in the article, and can't compute it from the test statistics, but the article reports a bount like p < <value>, return it here.
     */
    p_value_bound: (number | null);
};

export type FullExtractionResult = {
    inner: (FullExtractionResultSuccess | WrappedAbort);
};

export type FullExtractionResultSuccess = {
    tag?: 'success';
    claims: Claims;
    detailed_claim_results: Array<((ClaimDetailsSuccess | WrappedAbort) | null)>;
};

export type tag2 = 'success';

export type HTTPValidationError = {
    detail?: Array<ValidationError>;
};

export type PaperMD = {
    title: string;
    author_names: Array<(string)>;
    first_author: string;
    id: string;
    doi: ([
    string,
    string
] | null);
    type: (string | null);
};

export type ValidationError = {
    loc: Array<(string | number)>;
    msg: string;
    type: string;
};

export type WorkResponse = {
    md: PaperMD;
    images: Array<(string)>;
    full_response: FullExtractionResult;
};

export type WrappedAbort = {
    /**
     * The reason why the task is impossible to complete to complete and should be aborted.
     */
    reason: string;
    tag?: 'abort';
};

export type tag3 = 'abort';

export type GetImageImageWorkIdPageNumGetData = {
    path: {
        page_num: number;
        work_id: string;
    };
};

export type GetImageImageWorkIdPageNumGetResponse = (unknown);

export type GetImageImageWorkIdPageNumGetError = (HTTPValidationError);

export type GetAuthorsAuthorsGetResponse = (Array<[
    string,
    string,
    number
]>);

export type GetAuthorsAuthorsGetError = unknown;

export type GetAuthorAuthorAuthorIdGetData = {
    path: {
        author_id: string;
    };
    query?: {
        article_only?: boolean;
        first_author_only?: boolean;
    };
};

export type GetAuthorAuthorAuthorIdGetResponse = (AuthorResponse);

export type GetAuthorAuthorAuthorIdGetError = (HTTPValidationError);

export type GetWorkWorkWorkIdGetData = {
    path: {
        work_id: string;
    };
};

export type GetWorkWorkWorkIdGetResponse = (WorkResponse);

export type GetWorkWorkWorkIdGetError = (HTTPValidationError);