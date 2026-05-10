<?php

/**
 * Live-redirect entry point for the "Modern Dashboard" top-bar button.
 *
 * The button in `interface/main/tabs/main.php` used to compute its
 * `?pid=<uuid>` server-side at render time, which froze the URL on
 * whichever patient was open when the top tabs frame first loaded.
 * Subsequent patient switches updated `$_SESSION['pid']` (via
 * `set_pt.php` / `openpatient.php`) but never re-rendered the top
 * frame, so the link kept pointing at the original patient.
 *
 * This endpoint is hit on click instead. It reads the live session
 * pid, resolves the FHIR UUID, and 302s to the Modern Dashboard with
 * the correct patient. When no patient is open the redirect drops the
 * query string and the dashboard's own picker takes over.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Harrison Voegeli <harrison.voegeli@gmail.com>
 * @copyright Copyright (c) 2026 Harrison Voegeli
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

require_once('../globals.php');

use OpenEMR\Common\Database\QueryUtils;
use OpenEMR\Common\Session\SessionWrapperFactory;
use OpenEMR\Common\Uuid\UuidRegistry;

$copilotDashboardEnv = getenv('COPILOT_DASHBOARD_URL');
$copilotDashboardUrl = is_string($copilotDashboardEnv) ? trim($copilotDashboardEnv) : '';
if ($copilotDashboardUrl === '') {
    http_response_code(404);
    echo 'COPILOT_DASHBOARD_URL is not configured for this deployment.';
    exit;
}
$copilotDashboardUrl = rtrim($copilotDashboardUrl, '/');

$session = SessionWrapperFactory::getInstance()->getActiveSession();
$sessionPidValue = $session->get('pid');
$sessionPid = is_numeric($sessionPidValue) ? (int) $sessionPidValue : 0;

$patientUuid = '';
if ($sessionPid > 0) {
    $row = QueryUtils::querySingleRow("SELECT uuid FROM patient_data WHERE pid = ?", [$sessionPid]);
    $rawUuid = (is_array($row) && isset($row['uuid'])) ? $row['uuid'] : null;
    if (is_string($rawUuid) && $rawUuid !== '') {
        $patientUuid = UuidRegistry::uuidToString($rawUuid);
    }
}

$target = $copilotDashboardUrl . '/dashboard'
    . ($patientUuid !== '' ? ('?pid=' . urlencode($patientUuid)) : '');

header('Location: ' . $target);
exit;
