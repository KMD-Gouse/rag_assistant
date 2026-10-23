param(
    [Parameter(Mandatory = $false)]
    [string]$InputPath = "evaluation/results/benchmark-v2-development.json",

    [Parameter(Mandatory = $false)]
    [string]$OutputPath = "evaluation/results/benchmark-v2-development-per-query.csv"
)

$ErrorActionPreference = "Stop"

$report = Get-Content -LiteralPath $InputPath -Raw | ConvertFrom-Json
$diagnostics = @{}
foreach ($diagnostic in $report.diagnostics) {
    $diagnostics[$diagnostic.question_id] = $diagnostic
}

$rows = foreach ($result in $report.questions) {
    $diagnostic = $diagnostics[$result.question_id]
    $groupedEvidence = [ordered]@{}
    $ungroupedIndex = 0

    foreach ($evidence in $diagnostic.evidence) {
        if ($null -ne $evidence.group -and [string]$evidence.group -ne "") {
            $group = [string]$evidence.group
        }
        else {
            $group = "evidence_$ungroupedIndex"
            $ungroupedIndex++
        }

        if (-not $groupedEvidence.Contains($group)) {
            $groupedEvidence[$group] = @()
        }

        $groupedEvidence[$group] += "$($evidence.doc_id):p$($evidence.pdf_page)"
    }

    $goldEvidence = foreach ($group in $groupedEvidence.Keys) {
        "$group=" + ($groupedEvidence[$group] -join "|")
    }

    $retrievedPages = for ($index = 0; $index -lt $result.retrieved.Count; $index++) {
        $item = $result.retrieved[$index]
        "$($index + 1):$($item.payload.doc_id):p$($item.payload.pdf_page)"
    }

    $retrievedChunks = for ($index = 0; $index -lt $result.retrieved.Count; $index++) {
        $item = $result.retrieved[$index]
        "$($index + 1):$($item.payload.chunk_id)"
    }

    [pscustomobject][ordered]@{
        question_id = $result.question_id
        question_type = $result.type
        answerable = $result.answerable
        method = $result.method
        query = $result.query
        inferred_filters = if ($null -ne $diagnostic.inferred_filters) {
            $diagnostic.inferred_filters | ConvertTo-Json -Compress
        } else { "{}" }
        required_gold_groups = $groupedEvidence.Count
        accepted_gold_locations = @($diagnostic.evidence).Count
        gold_evidence = $goldEvidence -join ";"
        evidence_recall_at_5 = $result.metrics.evidence_recall
        document_recall_at_5 = $result.metrics.document_recall
        mrr_at_5 = $result.metrics.mrr
        all_evidence_hit_at_5 = $result.metrics.all_evidence_hit
        latency_ms = [math]::Round([double]$result.latency_ms, 4)
        retrieved_pages = $retrievedPages -join ";"
        retrieved_chunks = $retrievedChunks -join ";"
    }
}

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$rows | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding utf8
Write-Output "Exported $($rows.Count) rows to $OutputPath"
