<?php

/**
 * Patient deep-link entry point for the Modern Dashboard.
 *
 * The dashboard's "Open in classic OpenEMR" out-links route through
 * here so the user lands on the right patient regardless of their
 * current OpenEMR login state:
 *
 *   - Already authenticated: set the patient on the session and
 *     redirect to demographics.php — the classic dashboard view loads
 *     with the patient already selected.
 *   - Not authenticated: redirect to login.php with `patientID` on
 *     the query string. login.php passes that through to the form
 *     as a hidden input; main_screen.php's existing `$_POST['patientID']`
 *     branch (~line 446) auto-opens the patient tab post-login.
 *
 * Without this hop, a click on an out-link from the modern dashboard
 * would either (a) bounce through the OpenEMR login and lose the
 * `set_pid` from the original URL, leaving the user on OpenEMR's
 * generic home, or (b) require Co-Pilot to know the user's classic
 * OpenEMR session state — which it can't, because Co-Pilot and
 * classic OpenEMR are served from different cloudflared subdomains
 * and cannot share session cookies.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    Harrison Voegeli <harrison.voegeli@gmail.com>
 * @copyright Copyright (c) 2026 Harrison Voegeli
 * @license   https://github.com/openemr/openemr/blob/master/LICENSE GNU General Public License 3
 */

$ignoreAuth = true;
$sessionAllowWrite = true;
require_once('../globals.php');

use OpenEMR\Common\Session\PatientSessionUtil;
use OpenEMR\Common\Session\SessionWrapperFactory;

// FILTER_VALIDATE_INT rejects non-integer-looking input (leading zeros,
// trailing whitespace, scientific notation, floats) and yields false on
// failure. Coerce false to 0 so the next check catches it uniformly.
$pid = filter_input(INPUT_GET, 'pid', FILTER_VALIDATE_INT) ?: 0;
if ($pid <= 0) {
    http_response_code(400);
    echo 'pid query parameter must be a positive integer';
    exit;
}

$session = SessionWrapperFactory::getInstance()->getActiveSession();
$authUser = $session->get('authUser');

if (!is_string($authUser) || $authUser === '') {
    // Not authenticated. Carry the patient id through the login form so
    // main_screen.php opens the right chart tab after sign-in.
    header('Location: ../login/login.php?site=default&patientID=' . urlencode((string) $pid));
    exit;
}

// Authenticated. Set the patient on the session and route to the
// classic patient dashboard view. PatientSessionUtil also writes an
// audit-log entry for the view.
PatientSessionUtil::setPid($pid);
header('Location: ../patient_file/summary/demographics.php');
exit;
