jQuery( document).ready(function($){
	var copyid = 0;
	$('pre').each(function(){
		copyid++;
		$(this).attr('id', 'cb' + copyid).wrap( '<div class="pre-wrapper"/>');
		$(this).parent().css( 'margin', $(this).css( 'margin') );
		$('<button class="copy-snippet" data-clipboard-target="#cb'+copyid+'">Copy</button>').insertAfter( $(this) ).data( 'copytarget',copyid );
	});
});

var clipboard = new ClipboardJS('.copy-snippet');

clipboard.on('success', function(e) {
    e.trigger.textContent = 'Copied!';
    window.setTimeout(function() {
        e.trigger.textContent = 'Copy';
    }, 2000);
    e.clearSelection();

});

clipboard.on('error', function(e) {
    console.error('Action:', e.action);
    console.error('Trigger:', e.trigger);
    e.trigger.textContent = 'Failed';
    window.setTimeout(function() {
        e.trigger.textContent = 'Copy';
    }, 2000);
    e.clearSelection();
});
